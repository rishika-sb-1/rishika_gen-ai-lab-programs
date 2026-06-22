#!pip install langchain langchain-core langchain-community langchain-cohere langchain-text-splitters cohere pypdf faiss-cpu tiktoken sentence-transformers
import os
import warnings
warnings.filterwarnings("ignore")
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_cohere import ChatCohere
COHERE_API_KEY = "jAc2q1zxhfvtO3UJNab8Tftn6FheNlgOM8xDTs"
PDF_PATH = "IPC.pdf"
os.environ["COHERE_API_KEY"] = COHERE_API_KEY
print(f" PDF ready → {PDF_PATH}")
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(pages)
print(f" {len(pages)} pages → {len(chunks)} chunks")
print(" Creating embeddings - using free HuggingFace model ...")
print(" First run downloads the model ~90MB, takes 1-2 min ...")

embeddings = CohereEmbeddings(
    model="embed-english-v3.0"
)
vector_store = FAISS.from_documents(chunks, embeddings)
print(" Vector store ready")

prompt = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""
You are an Indian Penal Code legal assistant.

Answer the user's question only from the given context.

Use this format:

Section:
Mention the IPC section if available.

Explanation:
Explain the meaning in simple language.

Punishment:
Mention punishment if available.

If the answer is not available in the context, say:
"I could not find this clearly in the provided IPC PDF."

Context:
{context}

Question:
{question}

Chat History:
{chat_history}

Answer:
"""
)

llm = ChatCohere(
    model="command-a-03-2025",
    temperature=0.2
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}
)

chain = prompt | llm

def chat(question):
    docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    result = chain.invoke({
        "context": context,
        "question": question,
        "chat_history": ""
    })

    print("\nBot:")
    print(result.content)
    print()

print(" Chatbot ready!")
print("Type 'exit' to stop.\n")

while True:
    question = input("You: ")

    if question.lower() == "exit":
        break

    chat(question)