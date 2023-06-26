from langchain import LLMMathChain, OpenAI, SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from contants import open_ai_key, serp_api_key
import chainlit as cl
import os


os.environ['OPENAI_API_KEY'] = open_ai_key
os.environ['SERPAPI_API_KEY'] = serp_api_key

@cl.langchain_factory(use_async=False)
def load():
    llm = ChatOpenAI(temperature=0, streaming=True)
    llm1 = OpenAI(temperature=0, streaming=True)
    search = SerpAPIWrapper()
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

    tools = [
        Tool(
        name='Search',
        func=search.run,
        description='This is a text search engine inside the tools library'
        ),

        Tool(
        name='Calculator',
        func=llm_math_chain.run,
        description="This is a calculator that returns a solution to every math object."
        )
    ]
    return initialize_agent(
        tools, llm1, agent="chat-zero-shot-react-description", verbose=True
    )