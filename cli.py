from audioop import rms
from openai import OpenAI
from dotenv import main
import os, json, re
from argparse import ArgumentParser

from instructor import OpenAISchema
from typing import List
from pydantic import BaseModel, Field, conint, field_validator,confloat

main.load_dotenv()


GENERATE_PROMPT_ = """Your task is to write a checklist of the elements present in the provided text that a candidate text should have to be considered compatible with the reference text in terms of {criterion}. The checklist will be used to evaluate the quality of a candidate text.

### Evaluation Criteria
{criterion}: {criterion_definition}

You must follow the following rules when writing the checklist:

1. The checklist should contain yes/no questions.
2. Checklist items must be self-contained.
3. Focus on the main concepts and avoid including minor details.
4. Each item should represent a unique concept or element from the text.
5. Avoid repetition and overlapping of concepts in the checklist items.
6. The checklist should be comprehensive but not exhaustive, aiming for
clarity and brevity.
7. Generate as many items as you think are necessary
"""

GENERATE_PROMPT = """Your task is to create a comprehensive checklist outlining the elements a candidate text must contain to align with the reference text in terms of {criterion}. This checklist will serve as a tool for assessing the quality and compatibility of the candidate text with the reference text.

Consider the following definition of {criterion}: {criterion_definition}
"""
# better prompt
# GENERATE_PROMPT = """Write a list of the key elements for a candidate text to """

EVALUATE_PROMPT = """Your task is to evaluate if the candidate text attends to the elements in the checklist.

### Evaluation Criteria
{criterion}: {criterion_definition}

### Checklist

{checklist}
"""
#  The checklist was generated based on the following criteria:

default_criterion_definitions = {
    "consistency": "the factual alignment between the summary and the summarized source. A factually consistent summary contains only statements that are entailed by the source document.",
    "coherence": "Coherence refers to the overall quality that ensures sentences in a text build logically from one to the next, forming a well-structured and well-organized body of information on a given topic.",
    "relevance": "selection of important content from the source. The summary should include only important information from the source document. Annotators were instructed to penalize summaries that contained redundancies and excess information.",
    "fluency": "the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure."
}

class ChecklistItem(BaseModel):
    number: conint(ge=1) = Field(..., description="The item number")
    text: str = Field(..., description="The text of the checklist item")
    # importance: confloat(ge=0, le=1) = Field(1, description="The importance of the item in the checklist")
    

class Checklist(OpenAISchema):
    items: List[ChecklistItem] = Field(..., description="The checklist items")

    def to_markdown(self):
        markdown = "# Checklist\n"
        for i,item in enumerate(self.items):
            markdown += f"{i+1}: {item.text}\n"
        return markdown

class ChecklistResponseItem(BaseModel):
    item: conint(ge=1) = Field(..., description="Identifier for the checklist item.")
    # explanation: str = Field(None, description="A brief explanation of why the candidate did or did not contemplate the item.")
    isChecked: bool = Field(..., description="Indicates if the candidate contemplates the item.")
    
class ChecklistResponse(OpenAISchema):
    """The responses from the evaluation checklist."""
    items: List[ChecklistResponseItem] = Field(..., description="List of individual checklist item responses.")

    def call(self):
        results = []
        for item in self.items:
            results.append({
                "item": item.item,
                "contemplated":item.isChecked,
                # "reason":item.reason,
            })
        return results
    
    def score(self):
        return sum([item.isChecked for item in self.items])/len(self.items)


client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def call_model(messages, model, tools=None):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        max_tokens=2000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        seed=12,
        tool_choice={"type":"function","function":{"name":"Checklist"}}
    )

def extract_object(text:str):
    # return json.loads(text)
    obj = json.loads(re.search(r"\[.*\]", text, re.DOTALL).group())
    # sanitize the object (avoid keys different from 'text' in the checklist items)
    for item in obj:
        if 'text' not in item:
            keys = list(item.keys())
            for key in keys:
                if type(item[key]) == str:
                    item['text'] = item[key]
                    del item[key]
    return obj

# json_pattern = re.search(r'\[.*\]', text, re.DOTALL)



def generate_checklist(text:str, criterion:str = "consistency", criterion_definition:str = None, model:str = "gpt-3.5-turbo"):
    # check text is a file or string
    if os.path.isfile(text):
        with open(text, "r") as file:
            text = file.read()

    if criterion_definition is None:
        criterion_definition = default_criterion_definitions.get(criterion, "No definition provided.")

    tools = [{"type":"function","function":Checklist.openai_schema}]

    messages=[
        {"role": "system", "content": GENERATE_PROMPT.format(criterion=criterion, criterion_definition=criterion_definition)},
        {"role": "user", "content": f"### Reference text\n{text}"},
    ]

    chat_completion = call_model(messages, model,tools)
    try:
        return Checklist.from_response(chat_completion)
    except Exception as e:
        print(chat_completion.choices[0].message)
        messages.append({"role":"assistant", "tool_calls":chat_completion.choices[0].message.tool_calls})
        messages.append({
            "tool_call_id": chat_completion.choices[0].message.tool_calls[0].id,
            "role": "tool",
            "name": "Checklist",
            "content": str(e),
        })
        # messages.append({"role":"user", "content": str(e)})
        chat_completion = call_model(messages, model, tools)
        return Checklist.from_response(chat_completion)


def evaluate_checklist(checklist:Checklist, candidate_text:str, criterion:str = "consistency", criterion_definition:str = None, model:str = "gpt-3.5-turbo"):
    # check text is a file or string
    if os.path.isfile(candidate_text):
        with open(candidate_text, "r") as file:
            candidate_text = file.read()

    if criterion_definition is None:
        criterion_definition = default_criterion_definitions.get(criterion, "No definition provided.")

    tools = [{"type":"function","function":ChecklistResponse.openai_schema}]

    chat_completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": EVALUATE_PROMPT.format(criterion=criterion, criterion_definition=criterion_definition, checklist=checklist.to_markdown())},
            {"role": "user", "content": f"### Candidate text\n{candidate_text}"},
        ],
        tools=tools,
        temperature=0,
        tool_choice={"type":"function","function":{"name":"ChecklistResponse"}}
    )

    return ChecklistResponse.from_response(chat_completion)

if __name__ == "__main__":
    parser = ArgumentParser(description="Generate a checklist for a text summary.")
    parser.add_argument("--source_text", help="The text to generate the checklist for.")
    parser.add_argument("--candidate_text", help="The candidate text to evaluate.")
    parser.add_argument("--criterion", help="The criterion to generate the checklist for.", default="consistency")
    parser.add_argument("--criterion-definition", help="The definition of the criterion.")
    parser.add_argument("--model", help="The model to use for generating the checklist.", default="gpt-3.5-turbo")
    args = parser.parse_args()

    checklist = generate_checklist(args.source_text, args.criterion, args.criterion_definition, args.model)
    print(checklist.to_markdown())