from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser


def create_parser():
    parser = FlexibleArgumentParser()
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(model="meta-llama/Llama-3.2-1B-Instruct",
                        gpu_memory_utilization=0.8,
                        max_model_len=2048,
                        max_num_seqs=32)
    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument("--max-tokens", type=int)
    sampling_group.add_argument("--temperature", type=float)
    sampling_group.add_argument("--top-p", type=float)
    sampling_group.add_argument("--top-k", type=int)

    parser.add_argument("--system", type=str, default="You are a helpful assistant.")
    return parser

def main(args: dict):
    max_tokens = args.pop("max_tokens")
    temperature = args.pop("temperature")
    top_p = args.pop("top_p")
    top_k = args.pop("top_k")
    system = args.pop("system")

    llm = LLM(**args)

    sampling = llm.get_default_sampling_params()
    sampling.max_tokens = max_tokens
    if temperature is not None: sampling.temperature = temperature
    if top_p is not None: sampling.top_p = top_p
    if top_k is not None: sampling.top_k = top_k

    conversation = [{"role": "system", "content": system}]

    print("Chat ready. Type your message.")
    print("Commands: /reset (clear history), /exit (quit)\n")

    try:
        while True:
            user = input("You: ").strip()
            if not user:
                continue
            if user == "/exit":
                break
            if user == "/reset":
                conversation = [{"role": "system", "content": system}]
                print("History cleared.\n")
                continue

            conversation.append({"role": "user", "content": user})

            outputs = llm.chat(conversation, sampling, use_tqdm=False)
            reply = outputs[0].outputs[0].text
            print(f"Assistant: {reply}\n")

            conversation.append({"role": "assistant", "content": reply})

    except (KeyboardInterrupt, EOFError):
        print("\nBye!")

if __name__ == "__main__":
    parser = create_parser()
    args: dict = vars(parser.parse_args())
    main(args)