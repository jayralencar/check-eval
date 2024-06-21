from openai import OpenAI
import os, json
from dotenv import main
from instructor import OpenAISchema
from pydantic import Field
main.load_dotenv()

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

class Score(OpenAISchema):
    """The score of the evaluation."""
    score: float = Field(..., description="The score of the evaluation")

tools = [{"type":"function","function":Score.openai_schema}]

def call_model(messages, model, tools=None):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=2,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        n=20,
        tools=tools,
        tool_choice={"type":"function","function":{"name":"Score"}}
    )


def evaluate(source, system_output, criterion, model="gpt-4-turbo"):
    prompt = open(f"geval_prompts/{criterion}.md").read()
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"### Reference text\n{source}\n\n### System output\n{system_output}"},
    ]
    chat_completion = call_model(messages, model, tools=tools)
    scores = []
    Score.from_response(chat_completion)
    for choice in chat_completion.choices:
        # print(choice.message)
        scores.append(json.loads(choice.message.tool_calls[0].function.arguments)["score"])

    return sum(scores) / len(scores)
