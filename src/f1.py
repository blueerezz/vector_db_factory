# src/vectordb.py

# --- Dependencies ---
# Make sure you have these libraries installed:
# pip install chromadb sentence-transformers faiss-cpu numpy

import os
import pickle
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

# --- Abstract Base Class for Vector Databases ---
# This class defines the "contract" or interface that all our vector database
# implementations must follow. This is a core concept of modular design.
# It means you can swap out Chroma for Faiss (or any other DB you add later)
# and the rest of your code won't have to change.

class VectorDB(ABC):
    """
    Abstract base class for a Vector Database.
    """
    @abstractmethod
    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> None:
        """Adds texts and their embeddings to the database."""
        pass

    @abstractmethod
    def search(self, query_text: str, k: int = 3) -> List[Dict[str, Any]]:
        """Performs a similarity search and returns the top k results."""
        pass

    @abstractmethod
    def save(self) -> None:
        """Saves the database to disk."""
        pass

    @abstractmethod
    def load(self) -> None:
        """Loads the database from disk."""
        pass

# --- Concrete Implementation for ChromaDB ---
# Chroma is a full-featured vector database. It manages storage, indexing,
# and searching for you.

class ChromaDB(VectorDB):
    """
    ChromaDB implementation.
    """
    def __init__(self, db_path: str, collection_name: str, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        self.db_path = db_path
        self.collection_name = collection_name
        print("Initializing Sentence Transformer model for Chroma...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"} # Using cosine similarity
        )
        print("ChromaDB initialized.")

    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> None:
        if not texts:
            print("No texts to add.")
            return

        print(f"Embedding and adding {len(texts)} texts to Chroma...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        ids = [f"id_{i}" for i in range(len(self.collection.get()['ids']), len(self.collection.get()['ids']) + len(texts))]

        # ChromaDB requires metadata for each text, create empty ones if not provided
        if metadatas is None:
            metadatas = [{ "source": "unknown" } for _ in texts]

        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        print("Texts added to Chroma successfully.")

    def search(self, query_text: str, k: int = 3) -> List[Dict[str, Any]]:
        print(f"Searching for '{query_text}' in Chroma...")
        query_embedding = self.embedding_model.encode([query_text]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k
        )

        # Format the results to be consistent
        formatted_results = []
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                formatted_results.append({
                    'document': doc,
                    'distance': results['distances'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
                })
        return formatted_results

    def save(self) -> None:
        # Chroma with PersistentClient saves automatically.
        print("ChromaDB is persistent and saves automatically. No action needed for 'save'.")

    def load(self) -> None:
        # Chroma with PersistentClient loads automatically upon initialization.
        print("ChromaDB is persistent and loads automatically. No action needed for 'load'.")


# --- Concrete Implementation for FAISS ---
# FAISS (Facebook AI Similarity Search) is a library for efficient similarity search.
# It's not a full database, so we have to manage the storage of the actual text
# content ourselves. We'll use a simple list for that and save it with pickle.

class FaissDB(VectorDB):
    """
    FAISS implementation.
    """
    def __init__(self, db_path: str, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        self.db_path = db_path
        self.index_file = os.path.join(db_path, 'faiss.index')
        self.doc_store_file = os.path.join(db_path, 'doc_store.pkl')

        print("Initializing Sentence Transformer model for FAISS...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.doc_store: List[Dict[str, Any]] = [] # To store documents and metadata
        self.load()
        print("FAISS DB initialized.")

    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> None:
        if not texts:
            print("No texts to add.")
            return

        print(f"Embedding {len(texts)} texts for FAISS...")
        new_embeddings = self.embedding_model.encode(texts, show_progress_bar=True).astype('float32')

        if self.index is None:
            dimension = new_embeddings.shape[1]
            # Using IndexFlatL2, a basic but effective index for demonstration
            import faiss
            self.index = faiss.IndexFlatL2(dimension)

        self.index.add(new_embeddings)

        # Store the original texts and metadata
        if metadatas is None:
            metadatas = [{ "source": "unknown" } for _ in texts]

        for i, text in enumerate(texts):
            self.doc_store.append({'document': text, 'metadata': metadatas[i]})

        print("Texts added to FAISS successfully.")
        self.save() # Save after adding new data

    def search(self, query_text: str, k: int = 3) -> List[Dict[str, Any]]:
        if self.index is None or self.index.ntotal == 0:
            print("FAISS index is empty. Cannot perform search.")
            return []

        print(f"Searching for '{query_text}' in FAISS...")
        query_embedding = self.embedding_model.encode([query_text]).astype('float32')
        distances, indices = self.index.search(query_embedding, k)

        # Format the results
        formatted_results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1: # FAISS returns -1 for no result
                formatted_results.append({
                    'document': self.doc_store[idx]['document'],
                    'distance': float(distances[0][i]),
                    'metadata': self.doc_store[idx]['metadata']
                })
        return formatted_results

    def save(self) -> None:
        print(f"Saving FAISS index to {self.index_file}")
        os.makedirs(self.db_path, exist_ok=True)
        import faiss
        faiss.write_index(self.index, self.index_file)
        with open(self.doc_store_file, 'wb') as f:
            pickle.dump(self.doc_store, f)
        print("FAISS DB saved.")

    def load(self) -> None:
        if os.path.exists(self.index_file) and os.path.exists(self.doc_store_file):
            print(f"Loading FAISS index from {self.index_file}")
            import faiss
            self.index = faiss.read_index(self.index_file)
            with open(self.doc_store_file, 'rb') as f:
                self.doc_store = pickle.load(f)
            print("FAISS DB loaded successfully.")
        else:
            print("No existing FAISS DB found. A new one will be created upon adding data.")


# --- Vector Database Factory ---
# This is the factory you mentioned. Its job is to create the correct
# vector database object based on a simple string input. This hides the
# complexity of object creation from the main part of your application.

class VectorDBFactory:
    """
    Factory to create vector database instances.
    """
    @staticmethod
    def create_vectordb(db_type: str, db_path: str, collection_name: str = "default_collection") -> VectorDB:
        """
        Creates a vector database instance based on the specified type.

        Args:
            db_type (str): The type of database ('chroma' or 'faiss').
            db_path (str): The path to the database directory.
            collection_name (str): The name of the collection (for ChromaDB).

        Returns:
            VectorDB: An instance of a VectorDB subclass.
        """
        if db_type.lower() == 'chroma':
            return ChromaDB(db_path=db_path, collection_name=collection_name)
        elif db_type.lower() == 'faiss':
            return FaissDB(db_path=db_path)
        else:
            raise ValueError(f"Unknown vector database type: {db_type}")


# --- Example Usage ---
# This block demonstrates how to use the factory and the classes.
# It will only run when you execute this script directly (e.g., `python src/vectordb.py`).
# It won't run when you import this file into another script.

if __name__ == '__main__':
    # --- Configuration ---
    DB_TYPE = "faiss"  # <--- Change this to 'faiss' to test the other implementation
    DB_ROOT_PATH = "databases" # Folder to store all DBs
    DB_PATH = os.path.join(DB_ROOT_PATH, f"{DB_TYPE}_db")
    COLLECTION_NAME = "my_rag_collection"

    print(f"--- Running VectorDB Demo with '{DB_TYPE}' ---")

    # Create the database directory if it doesn't exist
    os.makedirs(DB_PATH, exist_ok=True)

    # 1. Use the factory to get the correct DB instance
    # Notice we don't need to know if it's Chroma or Faiss here, just that it's a VectorDB.
    vector_db = VectorDBFactory.create_vectordb(
        db_type=DB_TYPE,
        db_path=DB_PATH,
        collection_name=COLLECTION_NAME
    )

    # 2. Add some data (only if the DB is new/empty)
    # A simple check to avoid adding duplicate data on every run.
    # For a real app, you'd have a more robust way to check this.
    is_db_empty = True
    if DB_TYPE == 'chroma' and vector_db.collection.count() > 0:
        is_db_empty = False
    elif DB_TYPE == 'faiss' and vector_db.index is not None and vector_db.index.ntotal > 0:
        is_db_empty = False

    if is_db_empty:
        print("\nDatabase is empty. Adding sample documents...")
        sample_texts = [
            "The sky is blue.",
            "The sun is bright.",
            "Jupiter is the largest planet in our solar system.",
            "An apple a day keeps the doctor away.",
            "Python is a popular programming language."
        ]
        sample_metadatas = [
            {"source": "nature_facts.txt"},
            {"source": "nature_facts.txt"},
            {"source": "space_facts.txt"},
            {"source": "health_proverbs.txt"},
            {"source": "tech_docs.txt"}
        ]
        vector_db.add_texts(sample_texts, metadatas=sample_metadatas)
    else:
        print("\nDatabase already contains data. Skipping add.")


    # 3. Perform a similarity search
    print("\n--- Performing Searches ---")
    query1 = "What color is the sky?"
    results1 = vector_db.search(query1, k=3)
    print(f"\nQuery: '{query1}'")
    print("Results:")
    for res in results1:
        print(f"  - Document: {res['document']} (Distance: {res['distance']:.4f})")

    query2 = "Which programming language is widely used?"
    results2 = vector_db.search(query2, k=3)
    print(f"\nQuery: '{query2}'")
    print("Results:")
    for res in results2:
        print(f"  - Document: {res['document']} (Distance: {res['distance']:.4f})")

    print("\n--- Demo Finished ---")
