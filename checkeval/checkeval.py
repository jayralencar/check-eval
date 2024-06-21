from openai import OpenAI
import os, json
from instructor import OpenAISchema
from typing import List
from pydantic import BaseModel, Field, conint, field_validator,confloat
from pathlib import Path

default_criterion_definitions = {
    "consistency": "the factual alignment between the candidate and the reference text. A factually consistent candidate contains only statements that are entailed by the reference document.",
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

class Checkeval:
    def __init__(self, api_key, model="gpt-4-turbo"):
        self.model = model
        self.client = OpenAI(
            # This is the default and can be omitted
            api_key=api_key,
        )

    def call_model(self, messages, tools=None, tool_choice=None):
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            tools=tools,
            tool_choice=tool_choice,
            seed=10
        )
    
    def generate_checklist(self, criterion, text=None,  prompt=None):
        if prompt is None:
            # prompts is a directory containing the prompt files it is in the same level of this file inside the package
            path = Path(__file__).parent / "prompts" / "generate_checklist_pt.md"
            prompt = open(path).read()

            criterion_definition = default_criterion_definitions.get(criterion, "No definition provided.")

            prompt = prompt.format(criterion=criterion, criterion_definition=criterion_definition)
        
        tools = [{"type":"function","function":Checklist.openai_schema}]

        messages=[
            {"role": "system", "content": prompt}
        ]

        if text is not None:
            messages.append({"role": "user", "content": f"### Reference text\n{text}"})

        chat_completion = self.call_model(messages,tools, {"type":"function","function":{"name":"Checklist"}})
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
            chat_completion = self.call_model(messages,tools, {"type":"function","function":{"name":"Checklist"}})
            return Checklist.from_response(chat_completion)
    
    def evaluate_checklist(self, text, checklist, criterion, prompt=None):
        if prompt is None:
            # prompts is a directory containing the prompt files it is in the same level of this file inside the package
            path = Path(__file__).parent / "prompts" / "evaluate_checklist_pt.md"
            prompt = open(path).read()

            criterion_definition = default_criterion_definitions.get(criterion, "No definition provided.")

            if type(checklist) != str:
                checklist = checklist.to_markdown()

            prompt = prompt.format(criterion=criterion, criterion_definition=criterion_definition, checklist=checklist)
        
        tools = [{"type":"function","function":ChecklistResponse.openai_schema}]

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"### Candidate text\n{text}"},
        ]

        chat_completion = self.call_model(messages, tools, {"type":"function","function":{"name":"ChecklistResponse"}})

        return ChecklistResponse.from_response(chat_completion)

    def reference_guided(self, criterion, reference, candidate, checklist=None):
        
        if checklist is None:
            checklist = self.generate_checklist(criterion,reference)
        
        results = self.evaluate_checklist(candidate, checklist, criterion)

        return {
            "checklist":checklist,
            "results":results,
        }
        

    def candidate_guided(self, criterion, reference, candidate, checklist=None):
        return self.reference_guided(criterion, candidate, reference, checklist)

    def criterion_guided(self, criterion, reference, candidate, checklist=None):
        prompt = open(Path(__file__).parent / "prompts" / "criterion_generate_pt.md").read()
        criterion_definition = default_criterion_definitions.get(criterion, "No definition provided.")
        if checklist is None:
            prompt = prompt.format(criterion=criterion, criterion_definition=criterion_definition)

            checklist = self.generate_checklist(reference, criterion, prompt)

        if type(checklist) != str:
            checklist = checklist.to_markdown()
        
        evaluation_prompt = open(Path(__file__).parent / "prompts" / "criterion_evaluate_pt.md").read()

        evaluation_prompt = evaluation_prompt.format(criterion=criterion, criterion_definition=criterion_definition, checklist=checklist)

        messages = [
            {"role": "system", "content": evaluation_prompt},
            {"role": "user", "content": f"### Reference text\n{reference}\n\n### Candidate text\n{candidate}"},
        ]

        tools = [{"type":"function","function":ChecklistResponse.openai_schema}]

        chat_completion = self.call_model(messages, tools, {"type":"function","function":{"name":"ChecklistResponse"}})

        return {
            "checklist":checklist,
            "results":ChecklistResponse.from_response(chat_completion),
        }