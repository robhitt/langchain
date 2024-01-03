from langchain.document_loaders import TextLoader
from dotenv import load_dotenv

load_dotenv()

# rather than use Python's default file loader, we'll use the langchain
# langchain gives us many classes to load up a specific file type
loader = TextLoader("facts.txt")
docs = loader.load()

print(docs)
