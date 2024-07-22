import copy
from typing import List, Optional
import warnings
import torch
import torch.nn as nn
from traitlets import Callable
from transformers import PreTrainedModel, PretrainedConfig
from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from transformers import AutoTokenizer
from . import utils
from . import kv_cache
import importlib
importlib.reload(utils)
importlib.reload(kv_cache)
from .utils import *
from .kv_cache import initialize_past_key_values
import os
from huggingface_hub import hf_hub_download
import time
import sys

DEBUG = False

class ParagraphNode:
    def __init__(self, start: int, gid: int) -> None:
        self.start = start
        self.gid = gid
        self.termination = None

        self.next = None
        self.detail = None
    
    def as_dict(self):
        return {
            "start": self.start,
            "gid": self.gid,
            "termination": self.termination,
            "next": self.next.as_dict() if self.next is not None else None,
            "detail": self.detail.as_dict() if self.detail is not None else None,
        }

    @staticmethod
    def from_dict(dic):
        if dic is None:
            return None
        node = ParagraphNode(
            start=dic["start"],
            gid=dic["gid"],
        )
        node.termination = dic["termination"]
        node.set_next(ParagraphNode.from_dict(dic["next"]))
        node.set_detail(ParagraphNode.from_dict(dic["detail"]))       

    def set_next(self, next_node):
        self.next = next_node
    
    def set_detail(self, detail_node):
        self.detail = detail_node


    def _get_self_ids(self, global_input_ids, global_attention_masks, eop: int = 2):
        if self.termination is None:
            self_ids = global_input_ids[self.gid][self.start:]
            self_attn_mask = global_attention_masks[self.gid][self.start:]
        else:
            self_ids = global_input_ids[self.gid][self.start: self.termination]
            self_attn_mask = global_attention_masks[self.gid][self.start: self.termination]
        
        if not isinstance(self_ids, torch.Tensor):
            self_ids = torch.tensor(self_ids)
        if self_attn_mask.dtype != torch.bool:
            self_attn_mask = self_attn_mask.bool()
        
        real_ids = self_ids[self_attn_mask]
        real_ids = real_ids.tolist()
            
        if eop in real_ids:
            real_ids = real_ids[:real_ids.index(eop)]
        return real_ids

    def get_text(self, global_input_ids, global_attention_masks, tokenizer):
        """
        root - detail - next
        """
        assert (self.detail is not None) == (self.next is not None), f"detail and next should be both None or not None, but got detail: {self.detail}, next: {self.next}"

        text = ""

        # root
        self_ids = self._get_self_ids(global_input_ids, global_attention_masks)

        if DEBUG: print("self_ids", self.gid, self.start, self.termination, self_ids)

        if len(self_ids) != 0:
            if DEBUG: print(" > detokenzied self_ids", self_ids)
            text += tokenizer.decode(self_ids)
        
        # detail
        if self.detail is not None:
            detail_text = self.detail.get_text(global_input_ids, global_attention_masks, tokenizer)
            # HACK: force \n to fix the errors in training materials
            if not detail_text.startswith("\n"):
                detail_text = "\n" + detail_text.lstrip("\n")
            text += detail_text
        
        # next
        if self.next is not None:
            next_text = self.next.get_text(global_input_ids, global_attention_masks, tokenizer)
            # HACK: force \n to fix the errors in training materials
            if not next_text.startswith("\n\n"):
                next_text = "\n\n" + next_text.lstrip("\n")
            text += next_text

        return text
    
    def print_tree(self, depth: int = 0, tokenizer=None, global_input_ids=None, global_attention_masks=None):
        # if not DEBUG:
        #     return
        print(" >" * depth + f"ParagraphNode: gid {self.gid}, [{self.start}, {self.termination}) ", end="")
        if tokenizer is not None and global_input_ids is not None:
            self_ids = self._get_self_ids(global_input_ids, global_attention_masks)
            if len(self_ids):
                try:
                    print(repr(tokenizer.decode(self_ids)))
                except:
                    print("[decode error]")
                    import traceback
                    print(traceback.format_exc())
                    print(self_ids)
            else:
                print("[len = 0]")
        else:
            print()
        if self.detail is not None:
            self.detail.print_tree(depth + 1, tokenizer, global_input_ids, global_attention_masks)
        if self.next is not None:
            self.next.print_tree(depth, tokenizer, global_input_ids, global_attention_masks)


class MedusaConfig(PretrainedConfig):
    def __init__(
        self,
        medusa_num_heads=2,
        medusa_num_layers=1,
        base_model_name_or_path="lmsys/vicuna-7b-v1.3",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path


class ResBlock(nn.Module):
    """A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))


class MedusaModel(nn.Module):
    """The Medusa Language Model Head.

    This module creates a series of prediction heads (based on the 'medusa' parameter)
    on top of a given base model. Each head is composed of a sequence of residual blocks
    followed by a linear layer.
    """

    def __init__(
        self,
        base_model,
        medusa_num_heads=2,
        medusa_num_layers=1,
        base_model_name_or_path="lmsys/vicuna-7b-v1.3",
        medusa_head=None,
    ):
        """
        Args:
            base_model (nn.Module): The base language model to be used.
            medusa_num_heads (int, optional): Number of additional tokens to predict. Defaults to 3.
            medusa_num_layers (int, optional): Number of ResBlock layers for each Medusa head. Defaults to 0.
        """
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.medusa = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        if medusa_head is None:
            # Create a list of Medusa heads
            self.medusa_head = nn.ModuleList(
                [
                    nn.Sequential(
                        *([ResBlock(self.hidden_size)] * medusa_num_layers),
                        nn.Linear(self.hidden_size, self.vocab_size, bias=False),
                    )
                    for _ in range(medusa_num_heads)
                ]
            )

            # Ensure medusa_head's dtype and device align with the base_model
            self.medusa_head.to(self.base_model.dtype).to(self.base_model.device)

            for i in range(medusa_num_heads):
                # Initialize the weights of each medusa_head using the base model's weights
                self.medusa_head[i][-1].weight.data[:] = base_model.lm_head.weight.data[:]
        else:
            self.medusa_head = medusa_head

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
        cls,
        medusa_head_name_or_path,
        base_model=None,
        **kwargs,
    ):
        """
        Args:
            medusa_head_name_or_path (str): Name or path of the Medusa head to load.
            **kwargs: Additional keyword arguments for loading the base model.

        Returns:
            MedusaModel: A MedusaModel instance loaded from the given path.
        """
        medusa_config = MedusaConfig.from_pretrained(medusa_head_name_or_path)
        if base_model:
            print("Overriding base model as:", base_model)
            medusa_config.base_model_name_or_path = base_model
            
        base_model = KVLlamaForCausalLM.from_pretrained(
            medusa_config.base_model_name_or_path, **kwargs
        )
        model = cls(
            base_model,
            medusa_config.medusa_num_heads,
            medusa_config.medusa_num_layers,
            medusa_config.base_model_name_or_path,
        )
        medusa_head_path = os.path.join(medusa_head_name_or_path, "medusa_lm_head.pt")
        if os.path.exists(medusa_head_path):
            filename = medusa_head_path
        else:
            filename = hf_hub_download(medusa_head_name_or_path, "medusa_lm_head.pt")
        medusa_head_state_dict = torch.load(filename, map_location=base_model.device)
        model.medusa_head.load_state_dict(medusa_head_state_dict, strict=False)

        return model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
    ):
        """Forward pass of the MedusaModel.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            labels (torch.Tensor, optional): Ground truth labels for loss computation.
            past_key_values (tuple, optional): Tuple containing past key and value states for attention.
            output_orig (bool, optional): Whether to also output predictions from the original LM head.
            position_ids (torch.Tensor, optional): Position IDs.

        Returns:
            torch.Tensor: A tensor containing predictions from all Medusa heads.
            (Optional) Original predictions from the base model's LM head.
        """
        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
        # Clone the output hidden states
        hidden_states = outputs[0].clone()
        medusa_logits = []
        # TODO: Consider parallelizing this loop for efficiency?
        for i in range(self.medusa):
            medusa_logits.append(self.medusa_head[i](hidden_states))
        if output_orig:
            return torch.stack(medusa_logits, dim=0), outputs, orig
        return torch.stack(medusa_logits, dim=0)

    def medusa_generate(
        self,
        input_ids,
        attention_mask=None,
        temperature=0.0,
        max_steps=512,
        # The hyperparameters below are for the Medusa
        # top-1 prediciton for the next token, top-7 predictions for the next token, top-6 predictions for the next next token.
        medusa_choices=[1, 7, 6, 3],
        posterior_threshold=0.09,  # threshold validation of Medusa output
        # another threshold hyperparameter, recommended to be sqrt(posterior_threshold)
        posterior_alpha=0.3,
    ):
        """
        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            temperature (float, optional): Temperature for typical acceptance.
            medusa_choices (list, optional): A list of integers indicating the number of choices for each Medusa head.
            posterior_threshold (float, optional): Threshold for posterior validation.
            posterior_alpha (float, optional): Another threshold hyperparameter, recommended to be sqrt(posterior_threshold).
        Returns:
            torch.Tensor: Output token IDs.

        Warning: Only support batch size 1 for now!!
        """
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()

        # Cache medusa buffers (the fixed patterns for tree attention)
        if hasattr(self, "medusa_choices") and self.medusa_choices == medusa_choices:
            # Load the cached medusa buffer
            medusa_buffers = self.medusa_buffers
        else:
            # Initialize the medusa buffer
            medusa_buffers = generate_medusa_buffers(
                medusa_choices, device=self.base_model.device
            )
        self.medusa_buffers = medusa_buffers
        self.medusa_choices = medusa_choices

        medusa_topk = medusa_choices[1:]

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]

        reset_medusa_mode(self)
        # Initialize tree attention mask and process prefill token
        medusa_logits, logits = initialize_medusa(
            input_ids, self, medusa_buffers["medusa_attn_mask"], past_key_values
        )

        new_token = 0
        last_round_token = 0

        for idx in range(max_steps):
            # Generate candidates with topk predictions from Medusa heads
            # breakpoint()

            candidates, tree_candidates = generate_candidates(
                medusa_logits,
                logits,
                medusa_topk,
                medusa_buffers["tree_indices"],
                temperature,
            )

            # Use tree attention to verify the candidates and get predictions
            medusa_logits, logits, outputs = tree_decoding(
                self,
                tree_candidates,
                past_key_values,
                medusa_buffers["medusa_position_ids"],
                input_ids,
                medusa_buffers["retrieve_indices"],
            )

            # Evaluate the posterior of the candidates to select the accepted candidate prefix
            best_candidate, accept_length = evaluate_posterior(
                logits, candidates, temperature, posterior_threshold, posterior_alpha
            )

            print("evaluate_posterior returns:", best_candidate.item(), accept_length.item())

            # Update the input_ids and logits
            input_ids, logits, medusa_logits, new_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                medusa_buffers["retrieve_indices"],
                outputs,
                logits,
                medusa_logits,
                new_token,
                past_key_values_data,
                current_length_data,
            )

            yield {
                "text": self.tokenizer.decode(
                    input_ids[0, input_len:],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
            }

            if self.tokenizer.eos_token_id in input_ids[0, input_len:]:
                break

    def medusa_generate_batched(
        self,
        input_ids,  # (batch, len)
        attention_mask=None, # (batch, len)
        temperature=0.0,
        max_steps=512,
        # The hyperparameters below are for the Medusa
        # top-1 prediciton for the next token, top-7 predictions for the next token, top-6 predictions for the next next token.
        medusa_choices=[1, 7, 6, 3],
        posterior_threshold=0.09,  # threshold validation of Medusa output
        # another threshold hyperparameter, recommended to be sqrt(posterior_threshold)
        posterior_alpha=0.3,
    ):
        """
        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            temperature (float, optional): Temperature for typical acceptance.
            medusa_choices (list, optional): A list of integers indicating the number of choices for each Medusa head.
            posterior_threshold (float, optional): Threshold for posterior validation.
            posterior_alpha (float, optional): Another threshold hyperparameter, recommended to be sqrt(posterior_threshold).
        Returns:
            torch.Tensor: Output token IDs.
        """
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        batch_size = input_ids.shape[0]

        # Cache medusa buffers (the fixed patterns for tree attention)
        if hasattr(self, "medusa_choices") and self.medusa_choices == medusa_choices:
            # Load the cached medusa buffer
            medusa_buffers = self.medusa_buffers
        else:
            # Initialize the medusa buffer
            medusa_buffers = generate_medusa_buffers(
                medusa_choices, device=self.base_model.device
            )
        self.medusa_buffers = medusa_buffers
        self.medusa_choices = medusa_choices

        medusa_topk = medusa_choices[1:]

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model, batch_size=batch_size)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]

        print(input_ids.shape, attention_mask.shape)
        print("inputs:", input_ids.tolist(), attention_mask.tolist())

        reset_medusa_mode(self)
        # Initialize tree attention mask and process prefill tokens
        medusa_logits, logits = initialize_medusa(
            input_ids, attention_mask, self, medusa_buffers["medusa_attn_mask"], past_key_values
        )

        new_token = 0
        last_round_token = 0
        medusa_buffers["medusa_position_ids"] = medusa_buffers["medusa_position_ids"].expand(batch_size, -1).contiguous()
        next_position_id = torch.full((batch_size,), input_ids.shape[1], device=input_ids.device, dtype=torch.long)
        finished_position = torch.full((batch_size,), -1, dtype=torch.long, device='cpu')
        accept_length_store = [list() for _ in range(batch_size)]
        split_mask, eos = self.tokenizer.convert_tokens_to_ids(['[SplitMask]', '</s>'])
        terminal_tokens = {split_mask, eos}

        for idx in range(max_steps):
            # Generate candidates with topk predictions from Medusa heads

            candidates, tree_candidates = generate_candidates(
                medusa_logits,
                logits,
                medusa_topk,
                medusa_buffers["tree_indices"],
                temperature,
            )

            # Use tree attention to verify the candidates and get predictions
            medusa_logits, logits, outputs = tree_decoding(
                self,
                tree_candidates,
                attention_mask,
                past_key_values,
                medusa_buffers["medusa_position_ids"],
                input_ids,
                medusa_buffers["retrieve_indices"],
                next_position_id,
            )

            # Evaluate the posterior of the candidates to select the accepted candidate prefix
            best_candidate, accept_length = evaluate_posterior(
                logits, candidates, temperature, posterior_threshold, posterior_alpha
            )
            # print("evaluate_posterior inputs:", tuple(logits.shape), tuple(candidates.shape))
            print("evaluate_posterior returns:", best_candidate, accept_length)

            accept_length, gathered_candidates = postprocess_result(
                candidates, best_candidate, accept_length, terminal_tokens
            )

            previous_input_length = input_ids.shape[1]

            # Update the input_ids and logits
            (
                input_ids, 
                logits, 
                medusa_logits, 
                new_token,
                attention_mask,
                next_position_id,
            ) = update_inference_inputs(
                input_ids,
                gathered_candidates,
                best_candidate,
                accept_length,
                medusa_buffers["retrieve_indices"],
                outputs,
                logits,
                medusa_logits,
                new_token,
                past_key_values_data,
                current_length_data,
                attention_mask,
                next_position_id,
            )

            texts = []
            for batch_idx in range(batch_size):
                finished_position_index = finished_position[batch_idx].item()
                if finished_position_index == -1:
                    valid_tokens = input_ids[batch_idx, input_len:][attention_mask[batch_idx, input_len:].bool()]
                else:
                    valid_tokens = input_ids[batch_idx, input_len: finished_position_index][attention_mask[batch_idx, input_len: finished_position_index].bool()]

                if finished_position_index == -1:
                    if valid_tokens[-1] == eos:
                        finished_position[batch_idx] = previous_input_length + accept_length[batch_idx].item() + 1 # finished_position pointed to </s> + 1
                text = self.tokenizer.decode(valid_tokens)
                texts.append(text)
            yield {
                "text": texts
            }

            if (finished_position != -1).all():
                break

    @torch.inference_mode()
    def medusa_generate_apar(
        self,
        input_ids,  # (batch, len)
        attention_mask=None, # (batch, len)
        temperature=0.0,
        max_length=1024,
        max_depth=2,
        max_steps=-1, # obsolete
        # The hyperparameters below are for the Medusa
        # top-1 prediciton for the next token, top-7 predictions for the next token, top-6 predictions for the next next token.
        medusa_choices=[1, 7, 6],
        posterior_threshold=0.09,  # threshold validation of Medusa output
        # another threshold hyperparameter, recommended to be sqrt(posterior_threshold)
        posterior_alpha=0.3,
    ):
        """
        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            temperature (float, optional): Temperature for typical acceptance.
            medusa_choices (list, optional): A list of integers indicating the number of choices for each Medusa head.
            posterior_threshold (float, optional): Threshold for posterior validation.
            posterior_alpha (float, optional): Another threshold hyperparameter, recommended to be sqrt(posterior_threshold).
        Returns:
            torch.Tensor: Output token IDs.
        """
        # Avoid modifying the input_ids in-place
        original_input_ids = input_ids.cpu().clone()
        input_ids = input_ids.clone()
        batch_size, input_ids_seq_length = input_ids.shape
        
        if DEBUG:
            def dprint(*args, **kwargs):
                return print(*args, **kwargs)
        else:
            def dprint(*args, **kwargs):
                ...

        # Cache medusa buffers (the fixed patterns for tree attention)
        if hasattr(self, "medusa_choices") and self.medusa_choices == medusa_choices:
            # Load the cached medusa buffer
            medusa_buffers = self.medusa_buffers
        else:
            # Initialize the medusa buffer
            medusa_buffers = generate_medusa_buffers(
                medusa_choices, device=self.base_model.device
            )
        self.medusa_buffers = medusa_buffers
        self.medusa_choices = medusa_choices

        medusa_topk = medusa_choices[1:]

        # Calculate max batch size
        available_gpt_mem_gb = 25
        content_length = 3072
        single_batch_comsumption = self.base_model.config.num_hidden_layers * 2 * \
            content_length * \
            self.base_model.config.hidden_size * \
            2 # for 16bits type
        available_gpt_mem_bytes = available_gpt_mem_gb * (1 << 30)
        max_batch_size = available_gpt_mem_bytes // single_batch_comsumption
        print("Initializing KV Cache with", max_batch_size, "batches")

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model, batch_size=max_batch_size, content_length=content_length)  # -1 for max mem
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]

        print(input_ids.shape, attention_mask.shape)
        print("inputs:", input_ids.tolist(), attention_mask.tolist())

        reset_medusa_mode(self)
        # Initialize tree attention mask and process prefill tokens
        medusa_logits, logits = initialize_medusa(
            input_ids, attention_mask, self, medusa_buffers["medusa_attn_mask"], past_key_values
        )

        # === BEGIN OF BATCHED MEDUSA STORES ===
        new_token = 0
        next_position_id = torch.full((batch_size,), input_ids.shape[1], device=input_ids.device, dtype=torch.long)
        running_vector = torch.ones(batch_size, dtype=torch.bool, device='cpu')
        accept_length_store = [list() for _ in range(batch_size)]
        split_mask, split_sop, eos = self.tokenizer.convert_tokens_to_ids(['[SplitMask]', '[SplitSop]', '</s>'])
        terminal_tokens = {split_mask, eos}
        # === END OF BATCHED MEDUSA STORES ===

        # === BEGIN OF APAR STORES ===
        pending_insertion = {}
        finished_sequences = {}
        finished_masks = {}
        input_id_idx_to_global_idx = {i: i for i in range(batch_size)}
        global_idx_to_input_id_idx = {i: i for i in range(batch_size)}
        global_ptr = batch_size
        global_roots = [ParagraphNode(input_ids_seq_length, i) for i in range(batch_size)]
        global_idx_to_node = {i: global_roots[i] for i in range(batch_size)}  # the latest state of the tree node in the stream
        depth_counter = {i: 1 for i in range(batch_size)}
        max_running_batch_size = -1
        # === END OF APAR STORES ===
        # min_max_steps = (max_length - input_len) // (self.medusa + 1)
        # print("max steps supported", min_max_steps)
        
        while True:

            # print("next_position", next_position_id.shape, next_position_id.tolist())
            # print("input_ids", input_ids.shape, input_ids.tolist())
            # print("-----")
            if next_position_id.max().item() >= max_length:
                break

            if input_ids.shape[1] >= content_length:
                with open("warning.log", "a") as f:
                    print(">>> [beginning] Content length exceeded, next_position_id =", next_position_id.tolist(), file=f)
                    print("   Original_Input:", original_input_ids.tolist(), file=f)
                    print("    Exiting_Input:", input_ids.tolist(), file=f)
                break

            # dprint(" >>> begin of round", idx)
            # Generate candidates with topk predictions from Medusa heads

            if input_ids.shape[1] + self.medusa + 1 > max_batch_size > max_length:
                break

            # ban [SplitSop]
            medusa_logits[..., split_sop] = -1e3
            logits[..., split_sop] = -1e3

            # ban fork if reaching max_batch_size
            if batch_size == max_batch_size:
                dprint("[BAN] for max_batch_size is reached")
                medusa_logits[..., split_mask] = -1e3
                logits[..., split_mask] = -1e3
            
            # ban inputs that already reach max_fork
            for idx in range(input_ids.shape[0]):
                if depth_counter[input_id_idx_to_global_idx[idx]] >= max_depth:
                    dprint("[BAN] input_id_idx = %d, global_idx = %d" % (idx, input_id_idx_to_global_idx[idx]))
                    medusa_logits[:, idx, :, split_mask] = -1e3
                    logits[idx, :, split_mask] = -1e3

            candidates, tree_candidates = generate_candidates(
                medusa_logits, # [num_head, batch_size, input_len, vocab_size]
                logits, # [batch_size, input_len, vocab_size]
                medusa_topk,
                medusa_buffers["tree_indices"],
                temperature,
                pending_insertion, 
                global_idx_to_input_id_idx,
            )

            try:
                # Use tree attention to verify the candidates and get predictions
                medusa_logits, logits, outputs = tree_decoding(
                    self,
                    tree_candidates,
                    attention_mask,
                    past_key_values,
                    medusa_buffers["medusa_position_ids"].expand(batch_size, -1).contiguous(),
                    input_ids,
                    medusa_buffers["retrieve_indices"],
                    next_position_id,
                )
            except RuntimeError:
                print("HACK: context len exceeded!")
                import traceback
                with open("warning.log", "a") as f:
                    print(">>> [tree_decoding] Content length exceeded, next_position_id =", next_position_id.tolist(), file=f)
                    print("   Original_Input:", original_input_ids.tolist(), file=f)
                    print("    Exiting_Input:", input_ids.tolist(), file=f)
                    print(traceback.format_exc(), file=f)
                # raise
                break

            # Evaluate the posterior of the candidates to select the accepted candidate prefix
            best_candidate, accept_length = evaluate_posterior(
                logits, candidates, temperature, posterior_threshold, posterior_alpha
            )
            # print("evaluate_posterior inputs:", tuple(logits.shape), tuple(candidates.shape))
            dprint("evaluate_posterior returns:", best_candidate, accept_length)

            pending_insertion.clear()

            accept_length, gathered_candidates = postprocess_result(
                candidates, best_candidate, accept_length, terminal_tokens
            )

            previous_input_length = input_ids.shape[1]

            # Update the input_ids and logits
            (
                input_ids, 
                logits, 
                medusa_logits, 
                new_token,
                attention_mask,
                next_position_id,
            ) = update_inference_inputs(
                input_ids,
                gathered_candidates,
                best_candidate,
                accept_length,
                medusa_buffers["retrieve_indices"],
                outputs,
                logits,
                medusa_logits,
                new_token,
                past_key_values_data,
                current_length_data,
                attention_mask,
                next_position_id,
            )

            texts = []
            for batch_idx in range(batch_size):
                dprint("  <==>", batch_idx)
                valid_attention_mask = attention_mask[batch_idx].bool()
                valid_tokens = input_ids[batch_idx, input_len:][valid_attention_mask[input_len:]]

                text = self.tokenizer.decode(valid_tokens)
                texts.append(text)

                # Here, the next token has already been concatenated to the input_ids

                # APAR fork
                last_token = valid_tokens[-1].item()
                idx = batch_idx
                # print(">>> last token is", last_token, "idx is", idx, "global_ptr is", global_ptr, "global_idx_to_input_id_idx is", global_idx_to_input_id_idx)
                if last_token == split_mask:
                    dprint("[FORK]", idx)

                    # fork inputs
                    input_ids = torch.cat([input_ids, input_ids[idx: idx + 1]], dim=0)
                    attention_mask = torch.cat([attention_mask, attention_mask[idx: idx + 1]], dim=0)
                    medusa_logits = torch.cat([medusa_logits, medusa_logits[:, idx: idx + 1]], dim=1)
                    logits = torch.cat([logits, logits[idx: idx + 1]], dim=0)
                    next_position_id = torch.cat([next_position_id, next_position_id[idx: idx + 1]], dim=0)
                    running_vector = torch.cat([running_vector, running_vector[idx: idx + 1]], dim=0)

                    # HACK: fork past key values
                    past_key_values_data[:, batch_size, ...] = past_key_values_data[:, idx, ...]

                    # update pending and input_id_idx_to_global_idx
                    pending_insertion[global_ptr] = split_sop  # ids used between iterations and opreations should be global
                    input_id_idx_to_global_idx[len(input_ids) - 1] = global_ptr

                    # update node
                    for pos in range(len(input_ids[idx]) - 1, 0, -1):
                        if input_ids[idx][pos] == split_mask:
                            break
                    else:
                        raise ValueError("No split mask found in the input_ids")
                    split_mask_pos = pos
                    input_max_len = input_ids.shape[1]

                    # 1. create new nodes
                    next_node = ParagraphNode(input_max_len, input_id_idx_to_global_idx[idx])  # on the same stearm
                    detail_node = ParagraphNode(input_max_len + 1, global_ptr)

                    # 2. update current node
                    current_node = global_idx_to_node[input_id_idx_to_global_idx[idx]]
                    current_node.termination = split_mask_pos  # exlclude split mask
                    current_node.set_next(next_node)
                    current_node.set_detail(detail_node)

                    # 3. update global_idx_to_node
                    global_idx_to_node[input_id_idx_to_global_idx[idx]] = next_node
                    global_idx_to_node[global_ptr] = detail_node

                    # 4. create depth counter
                    depth_counter[global_ptr] = depth_counter[input_id_idx_to_global_idx[idx]] + 1

                    # update global ptr
                    global_ptr += 1
                    batch_size += 1
                # if not the only one
                elif last_token == eos:
                    dprint("[MARK TERM]", idx)
                    running_vector[idx] = 0

            max_running_batch_size = max(max_running_batch_size, batch_size)

            for idx, msk in enumerate(running_vector):
                if not msk:
                    # dprint("[Ending]", idx, repr(self.tokenizer.decode(input_ids[idx])))
                    finished_sequences[input_id_idx_to_global_idx[idx]] = input_ids[idx].cpu()
                    finished_masks[input_id_idx_to_global_idx[idx]] = attention_mask[idx].bool().cpu()
                else:
                    # update idx
                    new_idx = running_vector[:idx].int().sum().item()
                    if DEBUG: print(f"new_idx for <{idx}>", new_idx)
                    input_id_idx_to_global_idx[new_idx] = input_id_idx_to_global_idx[idx]
                    assert new_idx <= idx, f"asserting new_ids {new_idx} <= idx {idx}"

            global_idx_to_input_id_idx = {v: k for k, v in input_id_idx_to_global_idx.items() if k < running_vector.int().sum()}
            dprint("[Adjusted global ids]", global_idx_to_input_id_idx)
            dprint("[finished_sequences]", finished_sequences.keys())
            dprint("[depth_counter]", depth_counter)

            # batch reduction
            if not running_vector.all():
                running_batch_count = running_vector.int().sum().item()
                input_ids = input_ids[running_vector]
                attention_mask = attention_mask[running_vector]
                medusa_logits = medusa_logits[:, running_vector]
                logits = logits[running_vector]
                next_position_id = next_position_id[running_vector]
                past_key_values_data[:, :running_batch_count, :, :input_ids.shape[1]].copy_(
                    past_key_values_data[:, :batch_size][:, running_vector, :, :input_ids.shape[1]]
                )
                batch_size = running_vector.int().sum()

            ret_input_ids = []
            ret_attention_masks = []
            for idx in range(global_ptr):
                if idx in finished_sequences:
                    ret_input_ids.append(finished_sequences[idx])
                    ret_attention_masks.append(finished_masks[idx])
                else:
                    ret_input_ids.append(input_ids[global_idx_to_input_id_idx[idx]].cpu())
                    ret_attention_masks.append(attention_mask[global_idx_to_input_id_idx[idx]].bool().cpu())
            if DEBUG:
                for idx, r in enumerate(range(global_ptr)):
                    dprint(f"[Gid] {idx}", self.tokenizer.decode(ret_input_ids[idx]))
                
                for idx, r in enumerate(global_roots):
                    dprint(f"[Root] {idx}")
                    r.print_tree(
                        tokenizer=self.tokenizer, 
                        global_input_ids=ret_input_ids, 
                        global_attention_masks=ret_attention_masks
                    )

            yield global_roots, ret_input_ids, ret_attention_masks, global_idx_to_input_id_idx

            if running_vector.any():
                running_vector = running_vector[running_vector]
            else:
                break
        # else: 
        #     with open("warning.log", "a") as f:
        #         print(">>> WARNING: max_length reached", file=f)
        #         print("   Original_Input:", original_input_ids.tolist(), file=f)
        #         print("   Exiting_Input:", input_ids.tolist(), file=f)

    def medusa_trace(self, params: dict):
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(
            params['prompts'], 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=2048,
            return_attention_mask=True
        ).to(self.base_model.device)

        start_time = time.time()
        responses_sequence = []
        ret_input_ids_sequence = []
        global_root_sequence = []
        g_to_i_sequence = []
        elapse_sequence = []
        dynamic_bs = []

        for global_roots, ret_input_ids, ret_attention_masks, global_idx_to_input_id_idx in \
            self.medusa_generate_apar(
                **inputs, 
                max_length=params['max_length'],
                max_depth=params['max_depth'],
            ):
            if DEBUG: print(">>>>>>>>>>>>>>>>>>>")
            if DEBUG: print("[GLOBAL ROOTs]")
            for rid, r in enumerate(global_roots):
                if DEBUG: print("=====", rid)
                if DEBUG: r.print_tree(
                    tokenizer=self.tokenizer, 
                    global_input_ids=ret_input_ids, 
                    global_attention_masks=ret_attention_masks
                )

            responses = []
            for idx, r in enumerate(global_roots):
                response = r.get_text(ret_input_ids, ret_attention_masks, self.tokenizer)
                if DEBUG: print("[RET RESPONSE {}]".format(idx), response)
                responses.append(response)
            if DEBUG: sys.stdout.flush()
            if len(ret_input_ids) and isinstance(ret_input_ids[0], torch.Tensor):
                ret_input_ids = [ids.tolist() for ids in ret_input_ids]
            responses_sequence.append(responses)
            ret_input_ids_sequence.append(ret_input_ids)
            global_root_sequence.append([g.as_dict() for g in global_roots])
            g_to_i_sequence.append(global_idx_to_input_id_idx)
            elapse_sequence.append(time.time() - start_time)
            dynamic_bs.append(len(global_idx_to_input_id_idx))
        
        return {
            "responses": responses_sequence,
            "ret_input_ids": ret_input_ids_sequence,
            "global_root": global_root_sequence,
            "g_to_i": g_to_i_sequence,
            "elapse": elapse_sequence,
            "dynamic_bs": dynamic_bs,
        }