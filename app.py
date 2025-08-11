import os
import tempfile

import chromadb
import ollama
import streamlit as st
from typing import Tuple, List
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile

system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""


def process_document(uploaded_file: UploadedFile) -> list[Document]:
    """Read the uploaded PDF, split into chunks, and return Documents (Windows-safe)."""
    if uploaded_file is None:
        return []

    # Write uploaded file to a temp file, then CLOSE it before loading
    tmp = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
    try:
        tmp.write(uploaded_file.read())
        tmp.flush()
        tmp.close()  # Important for Windows so file is not locked

        # Load PDF content
        loader = PyMuPDFLoader(tmp.name)
        docs = loader.load()

    finally:
        # Remove temp file (best effort)
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    return text_splitter.split_documents(docs)



def get_vector_collection() -> chromadb.Collection:
    """Gets or creates a ChromaDB collection for vector storage.

    Creates an Ollama embedding function using the nomic-embed-text model and initializes
    a persistent ChromaDB client. Returns a collection that can be used to store and
    query document embeddings.

    Returns:
        chromadb.Collection: A ChromaDB collection configured with the Ollama embedding
            function and cosine similarity space.
    """
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text",
    )

    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )


def add_to_vector_collection(all_splits: list[Document], file_name: str):
    """Adds document splits to a vector collection for semantic search.

    Takes a list of document splits and adds them to a ChromaDB vector collection
    along with their metadata and unique IDs based on the filename.

    Args:
        all_splits: List of Document objects containing text chunks and metadata
        file_name: String identifier used to generate unique IDs for the chunks

    Returns:
        None. Displays a success message via Streamlit when complete.

    Raises:
        ChromaDBError: If there are issues upserting documents to the collection
    """
    collection = get_vector_collection()
    documents, metadatas, ids = [], [], []

    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    st.success("Data added to the vector store!")


def query_collection(prompt: str, n_results: int = 10):
    """Query vector store safely; handle empty store and clamp n_results."""
    collection = get_vector_collection()
    try:
        total = collection.count()
    except Exception:
        total = 0

    if total == 0:
        # return an empty-shaped result so downstream code won't crash
        return {"documents": [[]], "distances": [[]], "metadatas": [[]], "ids": [[]]}

    n = max(1, min(n_results, total))
    return collection.query(query_texts=[prompt], n_results=n)



def call_llm(context: str, prompt: str):
    """Calls the language model with context and prompt to generate a response.

    Uses Ollama to stream responses from a language model by providing context and a
    question prompt. The model uses a system prompt to format and ground its responses appropriately.

    Args:
        context: String containing the relevant context for answering the question
        prompt: String containing the user's question

    Yields:
        String chunks of the generated response as they become available from the model

    Raises:
        OllamaError: If there are issues communicating with the Ollama API
    """
    response = ollama.chat(
        model="llama3.2:3b",
        stream=True,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"Context: {context}, Question: {prompt}",
            },
        ],
    )
    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break


def re_rank_cross_encoders(query: str, documents: List[str]) -> Tuple[str, List[int]]:
    """Re-rank docs for a query; return concatenated top text + indices."""
    if not documents:
        return "", []

    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    top_k = min(3, len(documents))
    ranks = encoder_model.rank(query, documents, top_k=top_k)

    relevant_text = ""
    relevant_text_ids: List[int] = []
    for r in ranks:
        i = r["corpus_id"]
        relevant_text += documents[i].rstrip() + "\n\n"
        relevant_text_ids.append(i)

    return relevant_text.strip(), relevant_text_ids


if __name__ == "__main__":
    st.set_page_config(page_title="RAG Question Answer")  # <-- Move here

    # Document Upload Area
    with st.sidebar:
        uploaded_file = st.file_uploader(
            "**📑 Upload PDF files for QnA**", type=["pdf"], accept_multiple_files=False
        )
        process = st.button(
            "⚡️ Process",
        )
        if uploaded_file and process:
            normalize_uploaded_file_name = uploaded_file.name.translate(
                str.maketrans({"-": "_", ".": "_", " ": "_"})
            )
            all_splits = process_document(uploaded_file)
            add_to_vector_collection(all_splits, normalize_uploaded_file_name)

    # Question and Answer Area
    st.header("🗣️ RAG Question Answer")
    prompt = st.text_area("**Ask a question related to your document:**")
    ask = st.button(
        "🔥 Ask",
    )

if ask and prompt:
    collection = get_vector_collection()
    count = collection.count()
    if count == 0:
        st.warning("No documents indexed yet. Upload a PDF and click **Process** first.")
    else:
        results = query_collection(prompt, n_results=min(10, count))

        # Chroma returns a list per query; take the first safely
        docs_lists = results.get("documents", [])
        if not docs_lists or not docs_lists[0]:
            st.warning("No similar chunks found for your query. Try a broader question or re-process your docs.")
            with st.expander("See raw query results"):
                st.write(results)
        else:
            context_docs = docs_lists[0]  # list[str]

            # Re-rank with cross-encoder
            relevant_text, relevant_text_ids = re_rank_cross_encoders(prompt, context_docs)
            if not relevant_text:
                # fallback to top retrieved chunks if reranker returns nothing
                relevant_text = "\n\n".join(context_docs[:3])
                relevant_text_ids = list(range(min(3, len(context_docs))))

            # Generate answer
            response = call_llm(context=relevant_text, prompt=prompt)
            st.write_stream(response)

            with st.expander("See retrieved documents"):
                st.write(results)

            with st.expander("See most relevant document ids"):
                st.write(relevant_text_ids)
