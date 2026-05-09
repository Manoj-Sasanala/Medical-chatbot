from openai import OpenAI
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore


def load_pdf(data):

    loader = PyPDFLoader(data)

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )

    text_chunks = text_splitter.split_documents(documents)

    return text_chunks


def download_hugging_face_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


def store_embeddings(text_chunks, embeddings):

    vectorstore = PineconeVectorStore.from_documents(
        documents=text_chunks,
        embedding=embeddings,
        index_name="medical-chatbot"
    )

    return vectorstore

def search_query(query, embeddings):

    vectorstore = PineconeVectorStore(
        index_name="medical-chatbot",
        embedding=embeddings
    )

    results = vectorstore.similarity_search(query, k=3)

    return results


def generate_response(query, results):

    context = "\n".join([doc.page_content for doc in results])

    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )

    prompt = f"""
    You are a helpful medical assistant chatbot.

    Answer the user's question clearly and naturally using the provided medical context.

    Context:
    {context}

    Question:
    {query}
    """

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content