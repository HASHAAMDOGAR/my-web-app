import os
import openai
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from openai import azure_endpoint
from pydantic import BaseModel
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from openai import OpenAI

app = FastAPI()

azure_endpoint: str = "https://docwyn-ai-azure-sponsor.openai.azure.com"
azure_openai_api_key: str = "DQL6bnmdzu7oBNHwI26WXWz0MP3CAmWwmV8GnCpEj382CqX8bPAVJQQJ99ALACYeBjFXJ3w3AAABACOGwGZF"
azure_openai_api_version: str = "2023-05-15"
azure_deployment: str = "text-embedding-3-large"
azure_deployment_api: str = "DQL6bnmdzu7oBNHwI26WXWz0MP3CAmWwmV8GnCpEj382CqX8bPAVJQQJ99ALACYeBjFXJ3w3AAABACOGwGZF"

vector_store_address: str = "https://ragsearch2004.search.windows.net"
vector_store_password: str = "Vx0zm8JmWgvNElAWg5ycwpcL8EWmUEmwItMSc0DxmeAzSeB4JuQk"

client = OpenAI(
    base_url="http://172.210.62.81:4000/",
    api_key="sk-helloworldabcxyzlol1234"
)


embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
    azure_deployment=azure_deployment,
    openai_api_version=azure_openai_api_version,
    azure_endpoint=azure_endpoint,
    api_key=azure_openai_api_key,
)

index_name: str = "wine-db"
vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=vector_store_password,
    index_name=index_name,
    embedding_function=embeddings.embed_query,
)

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(
    "top_rated_wines.csv"
)
data = loader.load()

documents = loader.load()
#print(len(documents))

#for document in documents[100:200]:
 # print(vector_store.add_documents([document]))

class Body(BaseModel):
    query : str

@app.get('/')
def root():
    return RedirectResponse(url='/docs',status_code=301)

@app.post('/ask')
def ask(body: Body):
    """
    Use the query parameter to interact with the Azure OpenAI Service
    using the Azure Cognitive Search API for Retrieval Augmented Generation.
    """
    search_result = search(body.query)
    chat_bot_response = assistant(body.query, search_result)
    return {'response': chat_bot_response}



def search(query):
    """
    Send the query to Azure Cognitive Search and return the top result
    """
    docs = vector_store.similarity_search(
        query=query,
        k=3,
        search_type="similarity",
    )
    result = docs[0].page_content
    print(result)
    return result


def assistant(query, context):
    messages=[
        # Set the system characteristics for this chat bot
        {"role": "system", "content": "Asisstant is a chatbot that helps you find the best wine for your taste."},

        # Set the query so that the chatbot can respond to it
        {"role": "user", "content": query},

        # Add the context from the vector search results so that the chatbot can use
        # it as part of the response for an augmented context
        {"role": "assistant", "content": context}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-azure",
        messages=messages,
    )
    return str(response.__dict__)