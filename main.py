from prompt_demo import single_prompt_demo, chat_prompt_demo


def main():
    print("\nGemini Prompt Engineering (LangChain Integration)\n")
    # Show menu options to the user
    print("Choose an option:")
    print("1. PromptTemplate Workflow")
    print("2. ChatPromptTemplate Workflow")
    print("Type exit() to quit.\n")

    while True:
        choice = input("Select 1 or 2: ").strip()
        # Exit condition
        if choice.lower() in ["exit", "exit()", "quit"]:
            print("\nExiting the Prompt Demo...\n")
            break

        if choice == "1":
            single_prompt_demo()
        elif choice == "2":
            chat_prompt_demo()
        else:
            print("Invalid choice. Please select 1 or 2.\n")


if __name__ == "__main__":
    main()
