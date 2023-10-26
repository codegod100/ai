from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.document_loaders import DirectoryLoader
from langchain.chat_models.fireworks import ChatFireworks
from langchain.vectorstores import LanceDB
from langchain.chat_models import ChatOpenAI

import lancedb

from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)


# llm = ChatFireworks(model="accounts/fireworks/models/mistral-7b")
llm=ChatOpenAI(temperature=1)
db = lancedb.connect('.lance-data')
table = db.open_table('journal')
db = LanceDB(table, OpenAIEmbeddings())


vectorstore_info = VectorStoreInfo(
    name="flancian's journal",
    description="collection of markdown files containing flancian's daily journal",
    vectorstore=db,
)
query = "describe the pure land of flancia"

toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

agent_executor = create_vectorstore_agent(llm=llm, toolkit=toolkit, verbose=True)


resp = agent_executor.run(query)
print(resp)
