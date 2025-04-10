'''
Dummy agent library to demonstrate the use of the Hugging Face InferenceClient.
'''
import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


# Load the .env file from env file at the root of the project
dotenv_path = os.path.abspath(".env")
load_dotenv(dotenv_path, override=True)

HF_ACCESS_TOKEN = os.getenv("HF_TOKEN")

# This system prompt is a bit more complex and actually contains the function description already appended.
# Here we suppose that the textual description of the tools has already been appended.
SYSTEM_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

get_weather: Get the current weather in a given location

The way you use the tools is by specifying a json blob.
Specifically, this json should have an `action` key (with the name of the tool to use) and an `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are:
get_weather: Get the current weather in a given location, args: {"location": {"type": "string"}}
example use :

{{
  "action": "get_weather",
  "action_input": {"location": "New York"}
}}


ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about one action to take. Only one action at a time in this format:
Action:

$JSON_BLOB (inside markdown cell)

Observation: the result of the action. This Observation is unique, complete, and the source of truth.
... (this Thought/Action/Observation can repeat N times, you should take several steps when needed. The $JSON_BLOB must be formatted as markdown and only use a SINGLE action at a time.)

You must always end your output with the following format:

Thought: I now know the final answer
Final Answer: the final answer to the original input question

Now begin! Reminder to ALWAYS use the exact characters `Final Answer:` when you provide a definitive answer. """

if __name__ == "__main__":
    
    client = InferenceClient("meta-llama/Llama-3.2-3B-Instruct", token=HF_ACCESS_TOKEN)

    prompt=f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    {SYSTEM_PROMPT}
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    What's the weather in London ?
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

    output = client.text_generation(
        prompt,
        max_new_tokens=200,
        stop=["Observation:"] # Let's stop before any actual function is called
    )

    print(output)

    # Dummy weather function
    def get_weather(location: str) -> str:
        """ simulate a function that gets the weather in a given location """
        return f"the weather in {location} is sunny with low temperatures. \n"

    new_prompt = prompt + output + get_weather('London')
    
    final_output = client.text_generation(
        new_prompt,
        max_new_tokens=200,
    )

    print(final_output)
   