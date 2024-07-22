import json
from pprint import pprint as pp
import re

with open("sharegpt.jsonl", "r") as f:
    md_stru = [json.loads(line) for line in f.readlines()]

with open("parsed.json", "r") as f:
    para_ol_extraction = json.load(f)

unstructured_ids = set([x["id"] for x in para_ol_extraction])

filter_pattern = re.compile(r"```|<.*?>|\$|=|---|\d\d+ *[+x*×÷] *\d\d+")

code_response = []
for d in md_stru:
    dialogs = d['dialogs']
    if filter_pattern.search(dialogs[0]['response'].lower()):
        if d['id'] not in unstructured_ids:
            code_response.append(d)

print(len(code_response))
fastchat_resp = []
for item in code_response:
    conv = []
    for d in item['dialogs']:
        conv.append({
            "from": "human",
            "value": d['prompt']
        })
        conv.append({
            "from": "gpt",
            "value": d['response']
        })
    fastchat_resp.append({
        "id": item['id'],
        "conversations": conv
    })
print(len(fastchat_resp))
with open("unstructured.json", "w") as f:
    json.dump(fastchat_resp, f, indent=4)
