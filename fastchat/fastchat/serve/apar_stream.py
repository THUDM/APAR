from enum import Enum
import math
import copy
import os
import warnings
import re
import sys
from transformers.utils import logging
logger = logging.get_logger(__name__)

import torch
import torch.nn as nn
from transformers import GenerationMixin
from typing import Optional, Tuple, Union, List, Callable, Dict, Any
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig, ModelOutput
from transformers.generation.logits_process import SuppressTokensLogitsProcessor
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)


from transformers.generation.logits_process import LogitsProcessor
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.configuration_auto import AutoConfig

from argparse import ArgumentParser

import os
import torch
import argparse
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig, ModelOutput
from transformers.generation.logits_process import SuppressTokensLogitsProcessor
import json

from transformers import AutoModel, AutoTokenizer

import time

import fastchat
import importlib
importlib.reload(fastchat)
import fastchat.vs as vs
importlib.reload(vs)

stream_generate_tree_batch = vs.stream_generate_tree

# from fastchat.vs import stream_generate_tree as stream_generate_tree_batch
# import fastchat.vs as vs

# class ParagraphNode:
#     def __init__(self, start: int, gid: int) -> None:
#         self.start = start
#         self.gid = gid
#         self.termination = None

#         self.next = None
#         self.detail = None

#     def set_next(self, next_node):
#         self.next = next_node
    
#     def set_detail(self, detail_node):
#         self.detail = detail_node

#     def set_termination(self, termination):
#         self.termination = termination

#     def _get_self_ids(self, global_input_ids, eop: int = 2, retain_prefix: bool = False):
#         start_pos = 1 if retain_prefix else self.start  # only skip bos if retain_prefix
#         if self.termination is None:
#             self_ids = global_input_ids[self.gid][start_pos:]
#         else:
#             self_ids = global_input_ids[self.gid][start_pos:self.termination]
        
#         if isinstance(self_ids, torch.Tensor):
#             self_ids = self_ids.tolist()
            
#         if eop in self_ids:
#             # get the index of the right most eop
#             eop_index = len(self_ids) - self_ids[::-1].index(eop) - 1
#             self_ids = self_ids[:eop_index]
#         return self_ids

#     def get_text(self, global_input_ids, tokenizer, retain_prefix=False):
#         """
#         root - detail - next
#         """
#         assert (self.detail is not None) == (self.next is not None), f"detail and next should be both None or not None, but got detail: {self.detail}, next: {self.next}"

#         text = ""

#         # root, don't spread `retain_prefix`
#         self_ids = self._get_self_ids(global_input_ids, retain_prefix=retain_prefix)

#         # print("self_ids", self.gid, self.start, self.termination, self_ids)

#         if len(self_ids) != 0:
#             # print(" > detokenzied self_ids", self_ids)
#             text += tokenizer.decode(self_ids)
        
#         # detail
#         if self.detail is not None:
#             detail_text = self.detail.get_text(global_input_ids, tokenizer)
#             # HACK: force \n to fix the errors in training materials
#             if not detail_text.startswith("\n"):
#                 detail_text = "\n" + detail_text.lstrip("\n")
#             text += detail_text
        
#         # next
#         if self.next is not None:
#             next_text = self.next.get_text(global_input_ids, tokenizer)
#             # HACK: force \n to fix the errors in training materials
#             if not next_text.startswith("\n\n"):
#                 next_text = "\n\n" + next_text.lstrip("\n")
#             text += next_text

#         return text
    
#     def print_tree(self, depth: int = 0, tokenizer=None, global_input_ids=None):
#         print(" >" * depth + f"ParagraphNode: gid {self.gid}, [{self.start}, {self.termination}) ", end="")
#         if tokenizer is not None and global_input_ids is not None:
#             self_ids = self._get_self_ids(global_input_ids)
#             if len(self_ids):
#                 try:
#                     print(repr(tokenizer.decode(self_ids)))
#                 except:
#                     print("[decode error]")
#                     import traceback
#                     print(traceback.format_exc())
#                     print(self_ids)
#             else:
#                 print("[len = 0]")
#         else:
#             print()
#         if self.detail is not None:
#             self.detail.print_tree(depth + 1, tokenizer, global_input_ids)
#         if self.next is not None:
#             self.next.print_tree(depth, tokenizer, global_input_ids)

# @torch.inference_mode()
# def stream_generate_tree(
#         self,
#         input_ids,
#         split_mask: int,
#         split_sop: int,
#         generation_config: Optional[GenerationConfig] = None,
#         logits_processor: Optional[LogitsProcessorList] = None,
#         stopping_criteria: Optional[StoppingCriteriaList] = None,
#         prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
#         **kwargs,
# ):
#     batch_size, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]

#     if generation_config is None:
#         generation_config = self.generation_config
#     generation_config = copy.deepcopy(generation_config)
#     model_kwargs = generation_config.update(**kwargs)
#     bos_token_id, eos_token_id = generation_config.bos_token_id, generation_config.eos_token_id

#     if isinstance(eos_token_id, int):
#         eos_token_id = [eos_token_id]

#     print("eos_token_id", eos_token_id)

#     has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
#     if has_default_max_length and generation_config.max_new_tokens is None:
#         warnings.warn(
#             f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
#             "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
#             " recommend using `max_new_tokens` to control the maximum length of the generation.",
#             UserWarning,
#         )
#     elif generation_config.max_new_tokens is not None:
#         generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
#         if not has_default_max_length:
#             logger.warn(
#                 f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
#                 f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
#                 "Please refer to the documentation for more information. "
#                 "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)",
#                 UserWarning,
#             )

#     if input_ids_seq_length >= generation_config.max_length:
#         input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
#         logger.warning(
#             f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
#             f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
#             " increasing `max_new_tokens`."
#         )

#     # 2. Set generation parameters if not already defined
#     logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
#     stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

#     logits_processor = self._get_logits_processor(
#         generation_config=generation_config,
#         input_ids_seq_length=input_ids_seq_length,
#         encoder_input_ids=input_ids,
#         prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
#         logits_processor=logits_processor,
#     )

#     stopping_criteria = self._get_stopping_criteria(
#         generation_config=generation_config, stopping_criteria=stopping_criteria
#     )
#     logits_warper = self._get_logits_warper(generation_config)

#     print("basic config", logits_processor, stopping_criteria, logits_warper)

#     unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
#     scores = None


#     ###### HACK: ONLY SUPPORT BS = 1 ########
#     assert batch_size == 1, f"bs is expected to be 1, got {batch_size} instead"

#     pending_insertion = {}
#     finished_sequences = {}
#     input_id_idx_to_global_idx = {0: 0}
#     global_idx_to_input_id_idx = {0: 0}
#     global_ptr = 1
#     global_root = ParagraphNode(len(input_ids[0]), 0)
#     global_idx_to_node = {0: global_root}  # the latest state of the tree node in the stream
#     is_first = True
#     while True:
#         fork = False
#         model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

#         #### BEGIN OF DYNAMIC BATCH REDUCE
#         if model_inputs and len(model_inputs):
#             unfinished_mask = (unfinished_sequences == 1)
#             print("mask", unfinished_mask)
#             # print("input_ids before", model_inputs['input_ids'].shape)
#             model_inputs['input_ids'] = model_inputs['input_ids'][unfinished_mask]
#             # print("input_ids after", model_inputs['input_ids'].shape)

#             if model_inputs['past_key_values']:
#                 model_inputs['past_key_values'] = list(model_inputs['past_key_values'])
#                 for layer in range(len(model_inputs['past_key_values'])):
#                     # print("past_key_values before", model_inputs['past_key_values'][layer][0].shape)
#                     model_inputs['past_key_values'][layer] = (
#                         model_kwargs['past_key_values'][layer][0][unfinished_mask, ...],
#                         model_kwargs['past_key_values'][layer][1][unfinished_mask, ...],
#                     )
#                     # print("past_key_values after", model_inputs['past_key_values'][layer][0].shape)
#                 model_inputs['past_key_values'] = tuple(model_inputs['past_key_values'])
            
#             unfinished_sequences = unfinished_sequences[unfinished_mask]

#             # save se
#             for idx, msk in enumerate(unfinished_mask):
#                 if not msk:
#                     finished_sequences[input_id_idx_to_global_idx[idx]] = input_ids[idx]
#                 else:
#                     # update idx
#                     new_idx = unfinished_mask[:idx].int().sum().item()
#                     print(f"new_idx for <{idx}>", new_idx)
#                     input_id_idx_to_global_idx[new_idx] = input_id_idx_to_global_idx[idx]
#             # remove redundant idx
#             input_id_idx_to_global_idx = {k: v for k, v in input_id_idx_to_global_idx.items() if k < unfinished_mask.sum().item()}
#             input_ids = input_ids[unfinished_mask]

#         # check idx coverage
#         for i in range(global_ptr):
#             if i not in input_id_idx_to_global_idx.values() and i not in finished_sequences:
#                 print("missing idx", i)
#                 print("input_id_idx_to_global_idx", input_id_idx_to_global_idx)
#                 print("global_idx_to_input_id_idx", global_idx_to_input_id_idx)
#                 print("finished_sequences", finished_sequences)
#                 print("global_ptr", global_ptr)
#                 breakpoint()

#         # rebuild inverse mapping
#         global_idx_to_input_id_idx = {v: k for k, v in input_id_idx_to_global_idx.items()}

#         ### END OF DYNAMIC BATCH REDUCE

#         if not is_first:
#             # leave only the last token for each sequence
#             print("input_ids before", model_inputs['input_ids'].shape)
#             model_inputs['input_ids'] = model_inputs['input_ids'][:, -1:]
#         is_first = False

#         try:
#             outputs = self(
#                 **model_inputs,
#                 return_dict=True,
#                 output_attentions=False,
#                 output_hidden_states=False,
#             )
#         except:
#             import traceback
#             traceback.print_exc()
#             breakpoint()

#         next_token_logits = outputs.logits[:, -1, :]

#         # ban split_sop
#         next_token_logits[:, split_sop] = -1e3
#         # ban consecutive split_* token``
#         print("input shape", model_inputs["input_ids"].shape)
#         print("indexer", model_inputs["input_ids"][:, -1] == split_mask)
#         next_token_logits[model_inputs["input_ids"][:, -1] == split_mask, split_mask] = -1e3
#         next_token_logits[model_inputs["input_ids"][:, -1] == split_sop, split_mask] = -1e3

#         # pre-process distribution
#         next_token_scores = logits_processor(input_ids, next_token_logits)
#         next_token_scores = logits_warper(input_ids, next_token_scores)

#         # # TODO: this is incorrect ?
#         # # 2. ban split_mask on newly forked streams
#         # for idx in pending_ban_idx:
#         #     next_token_scores[global_idx_to_input_id_idx[idx], split_mask] = -float("inf")
#         # pending_ban_idx.clear()

#         # sample
#         probs = nn.functional.softmax(next_token_scores, dim=-1)

#         if generation_config.do_sample:
#             next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
#         else:
#             next_tokens = torch.argmax(probs, dim=-1)

#         print("next_tokens", next_tokens)

#         if 0 in next_tokens.tolist():
#             breakpoint()

#         # force pending tokens, note that we only support one pending token per stream
#         for idx, token in pending_insertion.items():
#             next_tokens[global_idx_to_input_id_idx[idx]] = token
#         pending_insertion.clear()

#         # update model states, put it here since it does not depend on the next token
#         model_kwargs = self._update_model_kwargs_for_generation(
#             outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
#         )


#         for idx in range(len(next_tokens)):
#             if next_tokens[idx].item() == split_mask and unfinished_mask[idx].item() == 1:
#                 print("[FORK]", idx)
#                 # HACK: fork next_tokens, input_ids and past KV
#                 next_tokens = torch.cat([next_tokens, next_tokens[idx: idx + 1]], dim=0)
#                 input_ids = torch.cat([input_ids, input_ids[idx: idx + 1]], dim=0)
#                 # HACK: fork past key values
#                 model_kwargs['past_key_values'] = list(model_kwargs['past_key_values'])
#                 for layer in range(len(model_kwargs['past_key_values'])):
#                     model_kwargs['past_key_values'][layer] = (
#                         torch.cat([model_kwargs['past_key_values'][layer][0], model_kwargs['past_key_values'][layer][0][idx: idx + 1,...]], dim=0),
#                         torch.cat([model_kwargs['past_key_values'][layer][1], model_kwargs['past_key_values'][layer][1][idx: idx + 1,...]], dim=0),
#                     )
#                 model_kwargs['past_key_values'] = tuple(model_kwargs['past_key_values'])
#                 unfinished_sequences = torch.cat([unfinished_sequences, unfinished_sequences[idx: idx + 1]], dim=0)

#                 # update pending and input_id_idx_to_global_idx
#                 pending_insertion[global_ptr] = split_sop  # ids used between iterations and opreations should be global
#                 input_id_idx_to_global_idx[len(input_ids) - 1] = global_ptr

#                 # update node
#                 split_mask_pos = len(input_ids[idx])
#                 split_sop_pos = split_mask_pos + 1

#                 # 1. create new nodes
#                 next_node = ParagraphNode(split_sop_pos, input_id_idx_to_global_idx[idx])  # on the same stearm
#                 detail_node = ParagraphNode(split_sop_pos + 1, global_ptr)

#                 # 2. update current node
#                 current_node = global_idx_to_node[input_id_idx_to_global_idx[idx]]
#                 current_node.termination = split_mask_pos  # exlclude split mask
#                 current_node.set_next(next_node)
#                 current_node.set_detail(detail_node)

#                 # 3. update global_idx_to_node
#                 global_idx_to_node[input_id_idx_to_global_idx[idx]] = next_node
#                 global_idx_to_node[global_ptr] = detail_node

#                 # update global ptr
#                 global_ptr += 1

#         # update generated ids
#         input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

#         unfinished_sequences = unfinished_sequences.mul((sum(next_tokens != i for i in eos_token_id)).long())

#         # stop when each sentence is finished, or if we exceed the maximum length
#         gen_eos = unfinished_sequences.max() == 0
#         sc_satisfy = stopping_criteria(input_ids, scores)
#         if gen_eos or sc_satisfy:
#             print("finish!", gen_eos, sc_satisfy)
#             break

#         # rebuild inverse mapping
#         global_idx_to_input_id_idx = {v: k for k, v in input_id_idx_to_global_idx.items()}

#         ret_input_ids = []
#         print("finished_sequences", finished_sequences.keys())
#         print("global_ptr", global_ptr)
#         print("input_id_idx_to_global_idx", input_id_idx_to_global_idx)
#         print("global_idx_to_input_id_idx", global_idx_to_input_id_idx)
#         print("==== end of iteration ====")
#         for idx in range(global_ptr):
#             if idx in finished_sequences:
#                 ret_input_ids.append(finished_sequences[idx])
#             else:
#                 ret_input_ids.append(input_ids[global_idx_to_input_id_idx[idx]])
#         yield global_root, ret_input_ids


@torch.no_grad()
def stream_chat_tree(self, tokenizer, params, device, context_len=2048, stream_interval=2):
    assert 'cuda' in device, f"device should be cuda, but got {device}"
    print("stream_chat_tree input", params)

    fastchat.vs.DEBUG = True

    prompt = params["prompt"]
    split_mask = tokenizer.encode("[SplitMask]")[1]
    split_sop = tokenizer.encode("[SplitSop]")[1]
    # assert split_mask > 150000 and split_sop > 150000, f"Possibly invalid encoding: split_mask: {split_mask}, split_sop: {split_sop}"
    logits_processor = LogitsProcessorList()
    # logits_processor.append(InvalidScoreLogitsProcessor())
    gen_kwargs = {"max_length": context_len, "do_sample": False, "logits_processor": logits_processor}

    inputs = tokenizer([prompt], return_tensors="pt", return_attention_mask=True)
    print("initial input tokens", inputs)
    inputs = inputs.to(self.device)
    yielded = False
    for i, (global_roots, ret_input_ids) in enumerate(stream_generate_tree_batch(self, **inputs, **gen_kwargs, split_mask=split_mask, split_sop=split_sop)):
        print(">>>>>>>>>>>>>>>>>>>")
        print("[GLOBAL ROOT]")
        global_root = global_roots[0]
        global_root.print_tree(tokenizer=tokenizer, global_input_ids=ret_input_ids)
        # print("[RET INPUT IDS]", ret_input_ids)
        # try:
        #     print("[RET INPUT TOKENS]", tokenizer.batch_decode(ret_input_ids))
        # except:
        #     for idx, input_id in enumerate(ret_input_ids):
        #         print(f"[RET INPUT TOKENS {idx}]", input_id)
        #         try:
        #             print(" >", tokenizer.decode(input_id))
        #         except:
        #             print(" > [ERROR]")
        response = global_root.get_text(ret_input_ids, tokenizer)
        # print("[RET prompt]", prompt)
        # print("[RET RESPONSE]", response)

        # response = self.process_response(response)
        sys.stdout.flush()
        yielded = False
        if i % stream_interval == 0:
            yielded = True
            yield response
    if not yielded:
        yield response

@torch.no_grad()
def get_trace(
    self, 
    tokenizer, 
    params: dict,
    logits_processor=None, 
    DEBUG=False,
):
    
    vs.DEBUG = DEBUG
    prompts = params["prompts"]
    split_mask = tokenizer.encode("[SplitMask]")[1]
    split_sop = tokenizer.encode("[SplitSop]")[1]
    # assert split_mask > 150000 and split_sop > 150000, f"Possibly invalid encoding: split_mask: {split_mask}, split_sop: {split_sop}"
    logits_processor = LogitsProcessorList()
    # logits_processor.append(InvalidScoreLogitsProcessor())
    gen_kwargs = {"max_length": params["max_length"], "do_sample": False, "logits_processor": logits_processor}

    inputs = tokenizer(prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=params["max_length"],
        return_attention_mask=True
    )
    print("initial input tokens", inputs)
    inputs = inputs.to(self.device)

    start_time = time.time()

    responses_sequence = []
    ret_input_ids_sequence = []
    global_root_sequence = []
    g_to_i_sequence = []
    elapse_sequence = []
    dynamic_bs = []

    for global_roots, ret_input_ids, global_idx_to_input_idx in vs.stream_generate_tree(self, 
        **inputs, **gen_kwargs, split_mask=split_mask, split_sop=split_sop, trace=True, 
        max_batch_size=params["max_batch_size"], max_depth=params["max_depth"]
    ):
        if DEBUG: print(">>>>>>>>>>>>>>>>>>>")
        if DEBUG: print("[GLOBAL ROOTs]")
        for rid, r in enumerate(global_roots):
            if DEBUG: print("=====", rid)
            if DEBUG: r.print_tree(tokenizer=tokenizer, global_input_ids=ret_input_ids)
        
        if DEBUG: 
            try:
                print("[RET INPUT TOKENS]", tokenizer.batch_decode(ret_input_ids))
            except:
                for idx, input_id in enumerate(ret_input_ids):
                    print(f"[RET INPUT TOKENS {idx}]", input_id)
                    try:
                        print(" >", tokenizer.decode(input_id))
                    except:
                        print(" > [ERROR]")

        responses = []
        for idx, r in enumerate(global_roots):
            response = r.get_text(ret_input_ids, tokenizer)
            if DEBUG: print("[RET RESPONSE {}]".format(idx), response)
            if getattr(self, "process_response", None):
                response = self.process_response(response)
            responses.append(response)
        if DEBUG: sys.stdout.flush()
        if len(ret_input_ids) and isinstance(ret_input_ids[0], torch.Tensor):
            ret_input_ids = [ids.tolist() for ids in ret_input_ids]
        responses_sequence.append(responses)
        ret_input_ids_sequence.append(ret_input_ids)
        global_root_sequence.append([g.as_dict() for g in global_roots])
        g_to_i_sequence.append(global_idx_to_input_idx)
        elapse_sequence.append(time.time() - start_time)
        dynamic_bs.append(len(global_idx_to_input_idx))

    return {
        "responses": responses_sequence,
        "ret_input_ids": ret_input_ids_sequence,
        "global_root": global_root_sequence,
        "g_to_i": g_to_i_sequence,
        "elapse": elapse_sequence,
        "dynamic_bs": dynamic_bs,
    }