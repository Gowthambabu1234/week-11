
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

# Use Gemini API key from environment variable
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7
)

# Prompt template
prompt = ChatPromptTemplate.from_template(
    "Write a very short story (2-3 sentences) about {topic}."
)

# Output parser
parser = StrOutputParser()

# LCEL chain
chain = prompt | llm | parser

# Hardcoded topics
topics = ["a dragon and a knight", "a lost spaceship"]

# Run and save the results
with open("output.txt", "w", encoding="utf-8") as file:
    for topic in topics:
        result = chain.invoke({"topic": topic})
        print(f"\nTopic: {topic}")
        print(f"Story: {result}")
        file.write(f"Topic: {topic}\nStory: {result}\n\n")
