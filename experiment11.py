
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load API key from .env
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")

MODEL = "openai/gpt-oss-120b"

# Initialize the language model
llm = ChatGroq(
    model=MODEL,
    groq_api_key=api_key,
    temperature=0.2
)

data = """RAG stands for Retrieval-Augmented Generation. It is an AI
framework that connects a Large Language Model (LLM) to external,
private, or real-time data sources. Instead of relying only on the
AI's memorized training data, RAG allows it to look up specific
information to provide more accurate and up-to-date answers."""

# Create the prompt template
prompt = ChatPromptTemplate.from_template(
    "Summarize the following text in exactly 10 sentences. "
    "Number each sentence from 1 to 10. "
    "Do not add an introduction or conclusion outside those sentences.\n\n"
    "{data}"
)

# Build the LangChain pipeline
chain = prompt | llm | StrOutputParser()

# Execute the chain
try:
    result = chain.invoke({"data": data})
    print("\nWEEK 11: RAG TEXT SUMMARIZATION")
    print("=" * 40)
    print(result)

    with open("output.txt", "w", encoding="utf-8") as file:
        file.write(result)

    print("\nOutput saved to output.txt")

except Exception as e:
    print("Error:", e)
