from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in one simple sentence."
)

model = ChatOpenAI(model="gpt-4o-mini")

output_parser = StrOutputParser()

# Create the chain
chain = prompt | model | output_parser

# Invoke the chain
response = chain.invoke({"topic": "Fire Alarm"})

print(response)