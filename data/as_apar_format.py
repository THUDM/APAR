import json
from tqdm import tqdm
import random
from argparse import ArgumentParser
from langdetect import detect
import re

MASK = "[SplitMask]"
STREAM_START = "[SplitSop]"

def en(text):
    try:
        language = detect(text)
        return language == 'en'
    except:
        return False

def pp(text):
    return text

def ch(text):
    return any('\u4e00' <= char <= '\u9fff' or '\u3040' <= char <= '\u309f' or '\uac00' <= char <= '\ud7a3' for char in text)

def pack_data(data_input):
    orig_exmaples = []
    examples = []
    for data in tqdm(data_input):
        for d in data['dialog'][:1]:
            if not d['forkable']:
                continue
            if ch(d['reassembled_response']):
                continue

            # confusing patterns
            if '```' in d['reassembled_response']:
                continue
            if '$$' in d['reassembled_response']:
                continue
            if '>' in d['reassembled_response']:
                continue
            if '<' in d['reassembled_response']:
                continue
            if '=' in d['reassembled_response']:
                continue
            if '---' in d['reassembled_response']:
                continue
            if 'http' in d['reassembled_response']:
                continue
            partitions = d['response'].split(MASK)
            assert len(d['streams']) + 1 == len(partitions), f"{len(d['streams'])} + 1 != {len(partitions)}"
            orig_exmaples.append({
                "index": data['id'],
                "input": pp(d['prompt']),
                "target": pp(d['reassembled_response'])
            })

            examples.append({
                "index": f"{data['id']}_main",
                "input": pp(d['prompt']),
                "target": pp(d['response'])
            })            
            for idx, stream in enumerate(d['streams']):
                examples.append({
                    "index": f"{data['id']}_{idx}",
                    "input": pp(d['prompt']),
                    "target": pp(MASK.join(partitions[:idx+1]) + MASK + STREAM_START + stream)
                })
    return examples, orig_exmaples

def filter_data(data_input):
    ret = []
    for data in tqdm(data_input):
        for d in data['dialog'][:1]:
            if not d['forkable']:
                continue
            ret.append(data)
    return ret

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--test_pct", type=float, default=0.12)
    parser.add_argument("--target_prefix", type=str, default="")

    args = parser.parse_args()

    with open("parsed.json", "r") as f:
        streaming_data = json.load(f)

    streaming_data = filter_data(streaming_data)
    print("Filtered data size: ", len(streaming_data))

    random.Random(100).shuffle(streaming_data)
    test_size = int(len(streaming_data) * args.test_pct)
    train, test = streaming_data[:-test_size], streaming_data[-test_size:]

    print("Train, Test =", len(train), len(test))

    with open(f"{args.target_prefix}test.json", "w") as f:
        json.dump(test, f, indent=4, ensure_ascii=False)
    with open(f"{args.target_prefix}train.json", "w") as f:
        json.dump(train, f, indent=4, ensure_ascii=False)

    packed_train, packed_orig = pack_data(train) 

    print(f"Train size: {len(packed_train)}")
    print(f"Orig size: {len(packed_orig)}")

    with open(f"{args.target_prefix}apar.json", "w") as f:
        json.dump(packed_train, f, indent=4, ensure_ascii=False)
    with open(f"{args.target_prefix}orig.json", "w") as f:
        json.dump(packed_orig, f, indent=4, ensure_ascii=False)