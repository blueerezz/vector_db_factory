from abc import ABC, abstractmethod
from typing import List, Optional

import os
import pickle
from abc import ABC
from typing import List, Dict, Any, Optional
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS


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

class ChromaDB(VectorDB):

    """
    ChromaDB implementation 
    """
    def __init__(self, db_path: str, collection_name: str, embedding_model_name: str = 'all-MiniLM-L6-v2'):

        self.db_path = db_path
        self.collection_name = collection_name
        print("Initializing Sentence Transformer model for Chroma...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.client =  chromadb.PersistentClient(path= self.db_path)
        self.collection = self.client.get_or_create_collection(
            name = self.collection_name , 
            metadata = {"hnsw:space": "cosine"}
        )
        print(" ChromaDB initialize.")

    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> None:
        if not texts:
            print("No texts to add.")
            return
        print(f"Embedding and adding {len(texts)} texts to Chroma...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        ids = [f"id_{i}" for i in range(len(self.collection.get()['ids']), len(self.collection.get()['ids']) + len(texts))]

        if metadatas is None:
             metadatas = [{ "source": "unknown" } for _ in texts]

        self.collection.add(
            embeddings= embeddings.tolist(), 
            documents= texts , 
            metadatas=  metadatas , 
            ids = ids
        )
        print("Texts added to Chroma successfully.")

    
    def search(self, query_text:str, k:int = 3) -> List[Dict[str, Any]]:
        print(f"Searching for '{query_text}' in Chroma...")
        query_embedding = self.embedding_model.encode([query_text]).tolist()
        results = self.collection.query(
            query_embeddings = query_embedding,
            n_results = k 
        )

        formatted_results = []

        if results["documents"]:
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
            #(brute‑force Euclidean index) 
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
    


