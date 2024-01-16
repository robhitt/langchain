from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI()
embeddings = OpenAIEmbeddings()

# We want to use the Chroma vector store in a specific directory and
# tell Chroma to use the OpenAIEmbeddings() class we created to vectorize the text
# Previously we used Chroma.from_documents() to create the vector store
db = Chroma(persist_directory="emb", embedding_function=embeddings)

# The retriever allows the langchain to be agnostic regardless of
# which Vector database we use
# A retriever (using method "get_relevant_documents") is an object
# that can take in a string and return some relevant list of documents
retriever = db.as_retriever()

# RetrievalQA is a class that wraps up the Chat Prompt Template
# retriever represents the vector store we created above
# "stuff" represents stuffing into the system message prompt template
chain = RetrievalQA.from_chain_type(llm=chat, retriever=retriever, chain_type="stuff")

result = chain.run("What is an interesting fact about the English language?")

print(result)
