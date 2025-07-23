import os
from src.vector_db2 import VectorDBFactory

def test_chroma():
    print("\n=== Testing ChromaDB ===")
    # point to persistent folder
    db = VectorDBFactory.create_vectordb(
        db_type="chroma",
        db_path="db/chroma",
        collection_name="demo"
    )

    # dummy texts + optional metadata
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "Lorem ipsum dolor sit amet",
        "Hello world from Chroma!"
    ]
    metadatas = [
        {"source": "sentence1"},
        {"source": "sentence2"},
        {"source": "sentence3"}
    ]

    # add and then search
    db.add_texts(texts, metadatas)
    results = db.search("quick fox", k=2)
    print("Chroma search results:", results)

def test_faiss():
    print("\n=== Testing FAISS ===")
    db = VectorDBFactory.create_vectordb(
        db_type="faiss",
        db_path="db/faiss"
    )

    texts = [
        "The quick brown fox jumps over the lazy dog",
        "Lorem ipsum dolor sit amet",
        "Hello world from FAISS!"
    ]
    # omit metadatas to test default
    db.add_texts(texts)

    results = db.search("hello world", k=2)
    print("FAISS search results:", results)

if __name__ == "__main__":
    # ensure folders exist
    os.makedirs("db/chroma", exist_ok=True)
    os.makedirs("db/faiss", exist_ok=True)

    test_chroma()
    test_faiss()
