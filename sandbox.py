# This was helpful: https://blog.gitguardian.com/how-to-handle-secrets-in-python/

# Set env var OPENAI_API_KEY or load from a .env file
import dotenv
dotenv.load_dotenv()




# ===== Loading

# from git import Repo
from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser

# Clone
repo_path = "D:/Libraries/Documents/GitHub/shapes"
# repo = Repo.clone_from("https://github.com/langchain-ai/langchain", to_path=repo_path)

# Load
loader = GenericLoader.from_filesystem(
    repo_path + "",
    glob="**/*",
    suffixes=[".py"],
    parser=LanguageParser(language=Language.PYTHON, parser_threshold=50),
)
documents = loader.load()
len(documents)


# ===== Splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=100, chunk_overlap=20,
)
texts = python_splitter.split_documents(documents)
len(texts)

if 1:
    for t in texts:
        print(t.page_content)


# ===== RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

if 0:
    # The example has this, but you need a paid plan for this not to violate
    #   the rate limit I guess.
    db = Chroma.from_documents(texts, OpenAIEmbeddings(disallowed_special=()))
else:
    # To avoid violating the rate limit.
    idx = 0
    chromadb.PersistentClient()
    lc_chroma = Chroma(client=persistent_client)
    db = Chroma.create_collection(name="shapes", embeddings=OpenAIEmbeddings(disallowed_special=()))
    for splitted_document in texts:
        print(f"Adding text {idx+1}... ",end="")
        db.add(documents=splitted_document)
        print("Done.")
        time.sleep(60)

retriever = db.as_retriever(
    search_type="mmr",  # Also test "similarity"
    search_kwargs={"k": 8},
)


# ===== Chat
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain

llm = ChatOpenAI(model_name="gpt-4")
memory = ConversationSummaryMemory(
    llm=llm, memory_key="chat_history", return_messages=True
)
qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)


# # Do the thing
# question = "What is the class hierarchy?"
# result = qa(question)
# result["answer"]
# print(result)




if __name__ == "__main__":
    print("NYI")