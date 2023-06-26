import os
from langchain import PromptTemplate, OpenAI, LLMChain
import chainlit as cl
from contants import open_ai_key

os.environ['OPEN_AI_KEY'] = open_ai_key

template = """ Question: {question}

Answer: Let's think about it on a colos level...
"""

@cl.langchain_factory(use_async=True)
def factory():
    prompt = PromptTemplate(template=template, input_variables=['question'])
    llm_chain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0), verbose=True)

    return llm_chain