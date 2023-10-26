from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.document_loaders import DirectoryLoader
from langchain.chat_models.fireworks import ChatFireworks

from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)


llm = ChatFireworks(model="accounts/fireworks/models/mistral-7b")
path = "/home/vera/agora/garden/flancian"
loader = DirectoryLoader(path, glob="**/*.md")
data = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
documents = text_splitter.split_documents(data)
db = FAISS.from_documents(documents, OpenAIEmbeddings())

vectorstore_info = VectorStoreInfo(
    name="flancian's journal",
    description="collection of markdown files containing flancian's daily journal",
    vectorstore=db,
)
query = "What does flancian think about Avalokiteśvara?"

toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

agent_executor = create_vectorstore_agent(llm=llm, toolkit=toolkit, verbose=True)


resp = agent_executor.run(query)
print(resp)
