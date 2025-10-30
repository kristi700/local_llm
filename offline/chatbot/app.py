import asyncio, uuid

from transformers import AutoTokenizer
from vllm import SamplingParams, AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

MODEL = "meta-llama/Llama-3.2-1B-Instruct"

async def render_prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

async def main():
    engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(
            model=MODEL,
            gpu_memory_utilization=0.8,
            max_model_len=2048,
            max_num_seqs=32,
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)

    sampling = SamplingParams(max_tokens=256, temperature=0.3)

    convo = [{"role": "system", "content": "You are a helpful assistant."}]
    print("Type /reset, /exit. Streaming enabled.\n")

    while True:
        try:
            user = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not user:
            continue
        if user == "/exit":
            break
        if user == "/reset":
            convo = [{"role": "system", "content": "You are a helpful assistant."}]
            print("History cleared.\n")
            continue

        convo.append({"role": "user", "content": user})
        prompt = await render_prompt(tokenizer, convo)

        rid = f"chat-{uuid.uuid4()}"
        print("Assistant: ", end="", flush=True)
        full = ""
        last_len = 0

        async for out in engine.generate(
            prompt=prompt,
            sampling_params=sampling,
            request_id=rid,
        ):
            cur = out.outputs[0].text
            print(cur[last_len:], end="", flush=True)
            last_len = len(cur)
            full = cur

        convo.append({"role": "assistant", "content": full})

if __name__ == "__main__":
    asyncio.run(main())
