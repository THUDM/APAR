import requests
import json
from pathlib import Path
from pprint import pprint as pp
from fastchat.conversation import get_conv_template
from argparse import ArgumentParser
import multiprocessing as mp
import os
from tqdm import tqdm
from functools import partial

data_root = Path(__file__).parent.parent.parent.parent / "data"

def mt_bench():
    with open(data_root / "mt_bench.jsonl") as f:
        return [json.loads(l) for l in f]
    

def vicuna_bench():
    with open(data_root / "vicuna_bench.jsonl") as f:
        return [json.loads(l) for l in f]


def _process(item, _args):
    resp_worker_addr = requests.post(
        _args.controller + "/get_worker_address",
        json={"model": _args.model}
    )
    resp_worker_addr = resp_worker_addr.json()
    worker_addr = resp_worker_addr['address']
    # print("Processing", item['question_id'], "@", worker_addr)
    traces = {
        **item,
        "model": _args.model,
        "traces": []
    }
    conv = get_conv_template(_args.conv_template)
    if _args.conv_template == 'llama-2':
        conv.set_system_message("You are a helpful, respectful and honest assistant.")
    for t in item['turns']:
        conv.append_message(conv.roles[0], t)
        conv.append_message(conv.roles[1], "")
        resp = query(
            formatted_prompts=conv.get_prompt(), 
            model=_args.model, 
            max_length=_args.max_length,
            worker_addr=worker_addr,
        )
        traces['traces'].append(resp)
        try:
            response = resp['responses'][-1][0]
            response_tokens = resp['ret_input_ids'][-1][0]
        except IndexError:
            response = ""
            response_tokens = []
        conv.messages = conv.messages[:-1]
        response = response.rstrip("</s>")
        conv.append_message(conv.roles[1], response)
    return traces

def query(
    formatted_prompts, 
    model, 
    worker_addr,
    max_length=50,
    max_depth=2,
    max_batch_size=32,
):
    worker_addr = worker_addr.rstrip("/")
    worker_addr = worker_addr + "/worker_trace"

    query_payload ={
        "prompts": formatted_prompts,
        "max_length": max_length,
        "max_batch_size": max_batch_size,
        "max_depth": max_depth,
        "model": model
    }

    resp = requests.post(
        worker_addr,
        json=query_payload
    )

    return resp.json()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', nargs='+', type=str)
    parser.add_argument('--max-length', type=int, default=1024)
    parser.add_argument('--save-path', type=str, required=True)
    parser.add_argument('--conv-template', type=str, default='vicuna_v1.1')
    parser.add_argument('--controller', type=str, required=True)

    args = parser.parse_args()
    args.controller = args.controller.rstrip("/")
    resp_worker_count = requests.post(args.controller + "/get_worker_count", json={"model": args.model})
    print("Getting worker count:", resp_worker_count, resp_worker_count.text)
    if resp_worker_count.status_code != 200:
        exit(1)
    resp_worker_count = resp_worker_count.json()
    args.num_workers = resp_worker_count['count']
    print(args.num_workers, "workers available")
    print("Proceeding with args:", args)

    root = Path(args.save_path)

    os.makedirs(root, exist_ok=True)
        
    if 'vicuna_bench' in args.dataset:
        print("Evaluating vicuna_bench")
        bench_data = vicuna_bench()
        with mp.Pool(args.num_workers) as p:
                results = list(tqdm(
                     p.imap_unordered(partial(_process, _args=args), bench_data),
                     total=len(bench_data)
                ))
                (root / "vicuna_bench.json").write_text(
                    json.dumps(results, ensure_ascii=False, indent=4)
                )

    if 'mt_bench' in args.dataset:
        print("Evaluating mt_bench")
        bench_data = mt_bench()
        with mp.Pool(args.num_workers) as p:
                results = list(tqdm(
                     p.imap_unordered(partial(_process, _args=args), bench_data),
                     total=len(bench_data)
                ))
                (root / "mt_bench.json").write_text(
                    json.dumps(results, ensure_ascii=False, indent=4)
                )