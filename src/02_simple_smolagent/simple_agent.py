'''
Simple ai agent using smolagents library.

This agent is designed to interact with the user and provide answers to their queries using the Qwen/Qwen2.5-Coder-32B-Instruct model.
It uses an LLM API to generate responses and can also utilize external tools for specific tasks.

WIP: This code is a work in progress and is being developed as part of a course on building AI agents.
'''
import os
import yaml

from dotenv import load_dotenv
from smolagents import CodeAgent, HfApiModel, load_tool
from tools.final_answer import FinalAnswerTool

# Load the .env file from env file at the root of the project
dotenv_path = os.path.abspath(".env")
load_dotenv(dotenv_path, override=True)

HF_ACCESS_TOKEN = os.getenv("HF_TOKEN")

def run_agent() -> None:
    """
    Main function to run the agent.
    """
    final_answer = FinalAnswerTool()

    model = HfApiModel(
        max_tokens=2096,
        temperature=0.5,
        model_id='Qwen/Qwen2.5-Coder-32B-Instruct', #it is possible that this model may be overloaded
        custom_role_conversions=None,
        token=HF_ACCESS_TOKEN,
    )

    # Import tool from Hub
    image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")

    with open("src/02_simple_smolagent/prompts.yaml", 'r') as stream:
        prompt_templates = yaml.safe_load(stream)

    agent = CodeAgent(
        model=model,
        tools=[final_answer, image_generation_tool], ## add your tools here (don't remove final answer)
        max_steps=6,
        verbosity_level=1,
        grammar=None,
        planning_interval=None,
        name=None,
        description=None,
        prompt_templates=prompt_templates
    )
    # Start CLI loop
    print("Agent ready. Type your prompt (Ctrl+C to quit):")
    while True:
        try:
            user_input = input("> ")
            response = agent.run(user_input)
            print(response)
        except KeyboardInterrupt:
            print("\nExiting.")
            break

if __name__ == "__main__":
    run_agent()