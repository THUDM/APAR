import argparse
import json
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

import argparse
import json
import os
from typing import Iterable, List
from pathlib import Path

from tqdm import tqdm
from conversation import get_conv_template
import asyncio
import aiohttp
import multiprocessing as mp
from datetime import datetime
import random
import time

engine = None



async def generate(request_dict) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream")
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    results_generator = engine.generate(prompt, sampling_params, request_id)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            prompt = request_output.prompt
            text_outputs = [
                prompt + output.text for output in request_output.outputs
            ]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\0").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt
    assert len(final_output.outputs) == 1
    output = final_output.outputs[0]
    ret = {
        "prompt": prompt,
        "text": output.text,
        **output.metadata,
    }
    return ret

async def issue_response(session, prompt: str) -> dict:
    pload = {
        "prompt": prompt,
        "n": 1,
        "use_beam_search": False,
        "temperature": 0.0,
        "max_tokens": 2048,
        "stream": False,
    }
    return await generate(pload)

async def main_issue_responses(prompts):
    async with aiohttp.ClientSession() as session:
        tasks = [issue_response(session, prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=Path, default=Path("../data/apar_testset.json"))
    parser.add_argument("--save-dir", type=Path, default=Path('vllm_results'))

    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    args.apar = True
    engine_args = AsyncEngineArgs.from_cli_args(args)

    date_string = str(datetime.now()).replace(" ", "-")
    args.save_dir.mkdir(exist_ok=True, parents=True)
    save_path = args.save_dir / f"responses_model_{Path(args.model).name}_prompt_{args.load.name}_time_{date_string}.json"
    engine_log_path = args.save_dir / f"engine_log_model_{Path(args.model).name}_prompt_{args.load.name}_time_{date_string}.jsonl"
    print("Responses will be saved to ", save_path)
    print("Engine logs will be saved to ", engine_log_path)

    engine = AsyncLLMEngine.from_engine_args(engine_args, log_file=engine_log_path)

    prompts = json.loads(args.load.read_text())
    random.Random(0).shuffle(prompts)
    print("Issuing", len(prompts), "prompts")
    start = time.time()
    responses = asyncio.run(main_issue_responses(prompts))
    elapse = time.time() - start

    save_path.write_text(json.dumps(responses, indent=4, ensure_ascii=False))
