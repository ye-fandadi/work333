from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv
import os

load_dotenv()

# def get_openai_chain():
#     """Builds an LLMChain using OpenAI's model"""
#     prompt = PromptTemplate.from_template("Answer this: {question}")
#     llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")
#     return LLMChain(llm=llm, prompt=prompt)

def get_deepseek_chain():
    """Builds an LLMChain using DeepSeek's model"""
    prompt = PromptTemplate.from_template("Answer this: {question}")
    llm = ChatDeepSeek(api_key=os.getenv("DEEPSEEK_API_KEY"), model="deepseek-chat")
    return LLMChain(llm=llm, prompt=prompt)

# def get_nvidia_chain():
#     """Builds an LLMChain using NVIDIA's DeepSeek-R1 model"""
#     prompt = PromptTemplate.from_template("Answer this: {question}")
#     llm = ChatNVIDIA(
#         api_key=os.getenv("NVIDIA_API_KEY"),
#         model="deepseek-ai/deepseek-r1",
#         temperature=0.6,
#         top_p=0.7,
#         max_tokens=4096
#     )
#     return LLMChain(llm=llm, prompt=prompt)

def get_chain_by_model(model_name: str) -> LLMChain:
    """Chooses the chain based on model selection"""

    if model_name == "deepseek":
        return get_deepseek_chain()
    else:
        raise ValueError("Unsupported model. Choose 'openai', 'deepseek', or 'nvidia'.")

if __name__ == "__main__":
    question = input("Enter your question: ").strip()
    model = input("Choose model (openai / deepseek / nvidia): ").strip().lower()
    chain = get_chain_by_model(model)

    response = chain.run({"question": question})
    print(f"\n[Response]:\n{response}")
