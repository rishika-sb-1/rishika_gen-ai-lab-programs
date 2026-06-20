# Program 10: Chatbot for Indian Penal Code

# !pip install requests PyMuPDF langchain faiss-cpu sentence-transformers numpy cohere

import requests
import fitz  # PyMuPDF
import os
from langchain.llms import Cohere
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

pdf_path = "ipc.pdf"
pdf_document = fitz.open(pdf_path)

ipc_text = ""

for page_num in range(pdf_document.page_count):
    page = pdf_document.load_page(page_num)
    ipc_text += page.get_text()

with open("IPC_text.txt", "w", encoding="utf-8") as text_file:
    text_file.write(ipc_text)

print("Text extracted and saved!")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_text(ipc_text)

model = SentenceTransformer("all-MiniLM-L6-v2")

document_embeddings = model.encode(texts, convert_to_tensor=True)

index = faiss.IndexFlatL2(document_embeddings.shape[1])
index.add(document_embeddings.cpu().numpy())

os.environ["COHERE_API_KEY"] = "Enter your Cohere API key here"

llm = Cohere(model="command-xlarge-nightly", temperature=0.7)

def get_chat_response(user_query):
    query_embedding = model.encode([user_query], convert_to_tensor=True)

    _, I = index.search(query_embedding.cpu().numpy(), k=1)

    most_similar_text = texts[I[0][0]]

    prompt = f"""
The user has asked a question related to the Indian Penal Code.
Below is the relevant section from the Indian Penal Code:

{most_similar_text}

The user's question: {user_query}

Please provide an answer based on the above IPC section.
"""

    response = llm(prompt)

    return response

user_input = input("Ask a question about the Indian Penal Code: ")

response = get_chat_response(user_input)

print(f"Chatbot Response: {response}")
