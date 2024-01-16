from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings

# from langchain.vectorstores import Chroma
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

# 1. Create a text splitter and its options.
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=200, chunk_overlap=0)

# rather than use Python's default file loader, we'll use the langchain
# langchain gives us many classes to load up a specific file type
# 2. Load text
loader = TextLoader("facts.txt")

# docs = loader.load()
# 3. Split text into chunks
docs = loader.load_and_split(text_splitter=text_splitter)

# 4. Embed chunks
# a) Load data, b) use OpenAI vectorization engine, c) create 'emb' vector store db
db = Chroma.from_documents(docs, embedding=embeddings, persist_directory="emb")

# 5. Search vector db based on query
results = db.similarity_search(  # returns results w/o the score
    # results = db.similarity_search_with_score(
    "What is an interesting fact about the English language?",
    # k=2,  # return 2 results
)

for result in results:
    print("\n")
    print(result.page_content)
