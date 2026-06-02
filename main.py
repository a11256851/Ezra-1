from src.agent import agent_manager


def main():
    thread_id = "user_1"
    print("LangChain Agent (Multi-turn with Memory)")
    print(f"Thread ID: {thread_id}")
    print("Type your questions or 'exit' to quit\n")

    try:
        while True:
            try:
                question = input("You: ").strip()
            except EOFError:
                print("\nGoodbye!")
                break

            if not question:
                continue

            if question.lower() in ["exit", "quit", "bye"]:
                print("Goodbye!")
                break

            print("Agent: ", end="", flush=True)

            try:
                for text_chunk in agent_manager.get_stream_response(
                    question, thread_id=thread_id
                ):
                    print(text_chunk, end="", flush=True)

            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)

            print()  # Add spacing between turns

    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
