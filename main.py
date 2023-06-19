# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
import os
os.environ["OPENAI_API_KEY"] = ""
reader = PdfReader('The Wonderful Wizard of Oz.pdf')
raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text
raw_text[:100]
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)

texts = text_splitter.split_text(raw_text)
len(texts)
embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_texts(texts, embeddings)

chain = load_qa_chain(OpenAI(), chain_type="stuff")

query = "what is Chapter I name of the article?"
docs = docsearch.similarity_search(query)
print(chain.run(input_documents=docs, question=query))