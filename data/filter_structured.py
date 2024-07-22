import json
import re
from tqdm import tqdm
import sys

MASK = "[SplitMask]"
STREAM_START = "[SplitSop]"

pattern = re.compile(r"((?:[\s\S]*\n\s*\d+\.\s+[\s\S]+[:：][\s\S]+\n\s*\d+\.\s+[\s\S]+[:：][\s\S]+\n\s*\d+\.\s+[\s\S]+[:：][\s\S]+.*)|(?:[\s\S]*\n\s*\-\s+[\s\S]+[:：][\s\S]+\n\s*\-\s+[\s\S]+[:：][\s\S]+\n\s*\-\s+[\s\S]+[:：][\s\S]+.*?))")

inter_num_pattern = re.compile(r"\s*(\d+\.)\s+([\s\S]+?)[:：]([\s\S]+?)(?=(?:\d+\.)\s+[\s\S]+?[:：])")
last_num_pattern = re.compile(r"\s*(\d+\.)\s+([\s\S]+?)[:：]([\s\S]+?)\n\n")
err_num_pattern = re.compile(r"\n\s*\d+\.\s+")

sentence_pattern = re.compile(r"(. |。)")

escaped_mask = re.escape(MASK)
split_mask_with_only_one_newline = re.compile(rf"({escaped_mask}\n)(?!\n)")

columned_data = []
uncolumned_data = []
all_data = []
streaming_data = []

splitter = re.compile(r"## ?(human|gpt)")


if __name__ == "__main__":

    all_data = {}
    # load filtered data
    with open("cleaned_sharegpt.jsonl", "r") as f:
        f_lines = f.readlines()
        for line in tqdm(f_lines, desc="loading dialogs"):
            data = json.loads(line)
            all_data[data["id"]] = data["dialogs"]
    print("all_md_data", len(columned_data), file=sys.stderr)

    annotated_data = []

    # extract main and streams
    for data_id, dialogs in tqdm(all_data.items(), total=len(all_data), desc='extracting ordered lists'):
        processed_dialogs = []
        for idx, data in enumerate(dialogs):
            streams = []
            current_idx = 0
            original_response = data["response"]
            prompt = data["prompt"]

            def process_m(m):
                global response
                indexer, summary, details = m
                global current_idx
                streams.append(details.rstrip("\n"))
                response = response.replace(details, f"{MASK}\n")

            try:
                response = data["response"] + "\n\n"
                matches = inter_num_pattern.findall(response)
                if matches:
                    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>")
                    pattern = re.compile(pattern)
                    for idx, m in enumerate(matches):
                        process_m(m)
                    # print("last responses ", response)
                    last_mask_pos = response.rfind(MASK)
                    last = last_num_pattern.findall(response[last_mask_pos + 1:])
                    if len(last) > 0:
                        process_m(last[0])
                    if any(map(lambda x: len(x) < 10, streams))\
                        or response.count(MASK) != len(streams) \
                        or len(streams) < 3\
                        or len(last) != 1\
                        or any(map(lambda x: err_num_pattern.search(x), streams)):
                        raise ValueError("illeagal match")

                    print("[NUM] <<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                    print("PROMPT: ", prompt)
                    print("ORIGINAL RESPONSE: ", data["response"])
                    print("RESPONSE: ", response)
                    print("STREAMS: ", streams)
                    print("STREAM LEN: ", len(streams))
                    # print("LAST LEN: ", len(last))

                    # ensure double \n as new line 
                    response = split_mask_with_only_one_newline.sub(f"{MASK}\n\n", response)

                    splitted_response = response.split(MASK)
                    assert len(splitted_response) == len(streams) + 1, f"len(splitted_response): {len(splitted_response)}, len(streams): {len(streams)}"
                    reassemled_response = splitted_response[0]
                    for s, r in zip(streams, splitted_response[1:]):
                        reassemled_response += s + r
                    processed_dialogs.append({
                        "prompt": prompt,
                        "response": response.rstrip(),
                        "original_response": original_response,
                        "reassembled_response": reassemled_response,
                        "forkable": True,
                        "streams": streams
                    })
                else:
                    raise ValueError("no match")
            except:
                try: # fallback to \n\n split
                    response = data["response"].strip()
                    paragraphs = list(filter(lambda x: x.strip(), response.split("\n\n")))
                    if len(paragraphs) <= 2:
                        raise ValueError("only 1 paragraph")
                    streams = []
                    processed = []
                    if '。' in response:
                        # Chinese
                        for p in paragraphs:
                            sentences = p.strip().split("。")
                            if len(sentences) <= 1:
                                processed.append(p)
                                continue
                            processed.append(sentences[0] + "。" + MASK)
                            streams.append("。".join(sentences[1:]).strip())
                    else:
                        # Others
                        for p in paragraphs:
                            sentences = p.strip().split(". ")
                            if len(sentences) <= 1:
                                processed.append(p)
                                continue
                            processed.append(sentences[0] + ". " + MASK)
                            streams.append(". ".join(sentences[1:]).strip())
                    if any(map(lambda x: len(x) < 20, processed)):
                        raise ValueError("maybe incorrect")
                    
                    response = "\n\n".join(processed)

                    lowered_resp = response.lower()

                    if "mr." in lowered_resp or "mrs." in lowered_resp or "dr." in lowered_resp:
                        raise ValueError("That would be an incorrect split")
                    
                    if len(streams) == 0:
                        raise ValueError("no stream")

                    print("[PAR] <<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                    print("PROMPT: ", prompt)
                    print("ORIGINAL RESPONSE: ", data["response"])
                    print("RESPONSE: ", response)
                    print("STREAMS: ", streams)
                    print("STREAM LEN: ", len(streams))

                    if response.count(MASK) < 3:
                        raise ValueError("To small paragraphs")

                    processed_dialogs.append({
                        "prompt": prompt,
                        "response": response.rstrip(),
                        "original_response": original_response,
                        "reassembled_response": original_response,
                        "forkable": True,
                        "streams": streams
                    })

                except ValueError:
                    processed_dialogs.append({
                        **data,
                        "forkable": False,
                        "streams": None
                    })

        assert len(processed_dialogs) == len(dialogs), f"len {len(processed_dialogs)} != {len(dialogs)}"
        if len(processed_dialogs) and any(map(lambda x: x['forkable'], processed_dialogs)):
            clean_dialogs(processed_dialogs)
            streaming_data.append({
                "id": data_id,
                "dialog": processed_dialogs
            })

    print("Streaming_data", len(streaming_data), file=sys.stderr)

    with open("parsed.json", "w") as f:
        json.dump(streaming_data, f, indent=4, ensure_ascii=False)

    streaming_data_ids = set()
    for data in streaming_data:
        streaming_data_ids.add(data['id'])
    print("Streaming_data_ids", len(streaming_data_ids), file=sys.stderr)
