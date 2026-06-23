#!pip install langchain cohere langchain-community
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Cohere
from langchain_community.document_loaders import TextLoader

cohere_api_key = "L"

file_path = "/content/sample text.txt"
loader = TextLoader(file_path)
documents = loader.load()

text_content = documents[0].page_content

prompt_template = PromptTemplate(
    input_variables=["text"],
    template="Analyze the following text and summarize its key points:\n\nText: {text}\n\nSummary:",
)

cohere_llm = Cohere(cohere_api_key=cohere_api_key, temperature=0.7)

chain = LLMChain(llm=cohere_llm, prompt=prompt_template)

output = chain.run(text=text_content)

print("Generated Summary:")
print(output)