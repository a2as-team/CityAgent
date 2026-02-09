from langchain_community.vectorstores import SupabaseVectorStore
import os
from src.rag_pipeline.vectorize_excel import vectorize_excel
from src.rag_pipeline.vectorize_pdf import vectorize_pdf
from src.ai_api_selector import get_embedding_model
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_TABLE_NAME = os.getenv("SUPABASE_VECTOR_TABLE", "chunks")
SUPABASE_QUERY_FN = os.getenv("SUPABASE_QUERY_FUNCTION", "match_chunks_langchain")


def get_supabase_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError(
            "Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY env vars. "
            "These must be set on the backend only."
        )
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


supabase_client = get_supabase_client()
embeddings = get_embedding_model()
vector_store = SupabaseVectorStore(
    client=supabase_client,
    embedding=embeddings,
    table_name=SUPABASE_TABLE_NAME,
    query_name=SUPABASE_QUERY_FN,
)


async def vectorize_file(filepath: str, file_key: str):
    """
    Vectorize a file into (documents, ids) using the existing vectorizers.

    Args:
        filepath (str): Path to the file to vectorize.
        file_key (str): Unique key for the file.

    Returns:
        tuple[list[Document], list[str]]: A tuple containing the list of
            `Document` objects and a parallel list of their string ids.
    """

    documents = []
    ids = []

    if filepath.endswith(".pdf"):
        documents, ids = await vectorize_pdf(filepath, file_key)
    elif filepath.endswith(".xlsx") or filepath.endswith(".csv"):
        documents, ids = await vectorize_excel(filepath, file_key)
    else:
        raise ValueError(f"Unsupported file type: {filepath}")

    return documents, ids


def add_documents_to_vector_store(documents, ids, chunk_size=5000):
    """Add documents to the vector store.

    Args:
        documents (list[Document]): List of Document objects to add.
        ids (list[str]): List of string ids corresponding to the documents.
    """
    for i in range(0, len(documents), chunk_size):
        print(
            f"Adding Document ID: {ids[i]} with content length: {len(documents[i].page_content)}"
        )
        chunk_docs = documents[i : i + chunk_size]
        chunk_ids = ids[i : i + chunk_size]
        print(f"Processing batch {i} to {i + len(chunk_docs)}...")
        vector_store.add_documents(documents=chunk_docs, ids=chunk_ids)


async def ingest_file(filepath: str, file_key: str) -> int:
    """
    Vectorize and insert all chunks for a file into Supabase pgvector.

    Args:
        filepath (str): Path to the file to ingest.
        file_key (str): Unique key for the file.

    Returns:
        int: Number of chunks ingested.
    """
    documents, ids = await vectorize_file(filepath, file_key)
    if not documents:
        return 0
    add_documents_to_vector_store(documents, ids)
    return len(documents)


def delete_file_chunks(file_key: str) -> int:
    """
    Delete all chunks from Supabase where metadata contains {"file_key": file_key}.

    Returns:
        number of deleted rows if available, else 0
    """

    res = (
        supabase_client.table(SUPABASE_TABLE_NAME)
        .delete()
        .contains("metadata", {"file_key": file_key})
        .execute()
    )

    # supabase-py returns data in res.data when available
    try:
        return len(res.data) if res.data else 0
    except Exception:
        return 0


def query_retriever(query: str, k: int = 4):
    """Run a query against the persisted vector store retriever.

    Args:
        query (str): The natural-language query to run.
        k (int): Number of top similar documents to retrieve.

    Returns:
        Any: The retriever's raw response (depends on configured retriever).
    """

    return vector_store.similarity_search(query, k=k)
