import asyncio, uuid

from transformers import AutoTokenizer
from vllm import SamplingParams, AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
MAX_HISTORY_SIZE = 3

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

    sampling = SamplingParams(max_tokens=256, temperature=0.3) # TODO top_p, outputkind etc

    convo = [{"role": "system", "content": "You are a helpful assistant."}]
    summary = ""
    convo = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
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
            summary = ""
            print("History cleared.\n")
            continue

        conversation_with_summary = convo.copy()
        if summary:
            conversation_with_summary.insert(
                1, {"role": "system", "content": f"Summary of previous conversation: {summary}"}
            )

        conversation_with_summary.append({"role": "user", "content": user})
        prompt = await render_prompt(tokenizer, conversation_with_summary)

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

        convo.append({"role": "user", "content": user})
        convo.append({"role": "assistant", "content": full})

        # stg is wrong /w summarization - need to get debugging work / vllm
        if len(convo) > MAX_HISTORY_SIZE * 2:
            messages_to_summarize = []
            for m in convo:
                if m["role"] in ("user", "assistant"):
                    messages_to_summarize.append(m)
            summary_prompt = f"Summarize the following conversation:\n"
            for m in messages_to_summarize[-MAX_HISTORY_SIZE*2:]:
                summary_prompt += f"{m['role'].capitalize()}: {m['content']}\n"
            summary_text = ""
            async for summary_out in engine.generate(
                prompt=summary_prompt,
                sampling_params=sampling,
                request_id=f"summary-{uuid.uuid4()}",
            ):
                summary_text = summary_out.outputs[0].text
            summary = summary_text.strip()
            system_msgs = [m for m in convo if m["role"] == "system"]
            dialog_msgs = [m for m in convo if m["role"] in ("user", "assistant")]
            dialog_msgs = dialog_msgs[-MAX_HISTORY_SIZE*2:]
            convo = system_msgs + dialog_msgs

if __name__ == "__main__":
    asyncio.run(main())
