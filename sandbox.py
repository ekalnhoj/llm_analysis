# This was helpful: https://blog.gitguardian.com/how-to-handle-secrets-in-python/

#%%
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

if 1:
    # The example has this, but you need a paid plan for this not to violate
    #   the rate limit I guess.
    db = Chroma.from_documents(texts, OpenAIEmbeddings(disallowed_special=()))
else:
    # From https://python.langchain.com/docs/modules/data_connection/text_embedding/
    from langchain.embeddings import OpenAIEmbeddings
    embeddings_model = OpenAIEmbeddings()

    embeddings = embeddings_model.embed_documents(
        [
            "Hi there!",
            "Oh, hello!",
            "What's your name?",
            "My friends call me World",
            "Hello World!"
        ]
    )
    len(embeddings), len(embeddings[0])

    embedded_query = embeddings_model.embed_query("What was the name mentioned in the conversation?")
    print(embedded_query[:5])

    pass


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

#%%
# Do the thing
question = ...
"""
For each class in the code, show me the class hierarchy following these rules:
- each class on its own line
- if class "A" has no parents, have the contents of the line be "A"
- if class "B" inherits from "A", have the contents of the line be "A -> B"
- do this recursively, so if E inherits from B which inherits from A, the contents of the line are "A -> B -> E"
"""

#%%
result = qa(question)
# result["answer"]
print("Result obtained!")

#%%
print(result["answer"])

#%%
question = \
"""
Give me documentation for each function in the code, including:
- arguments
- returns
- other actions

Put each documentation block in the example format below, with <> used to signify 
what type of info should be in each spot. Format starts here:
# ========== #
# <class name (if applicable)> : <function name>
# Arguments: <arguments>
# Returns: <returns>
# Actions: <actions>
# Notes: <any other info>
# ========== #
"""

if __name__ == "__main__":
    print("Code complete!")
# %%
