# Vector Database Abstraction with ChromaDB and FAISS

This project provides a unified and clean abstraction layer for using popular vector databases, specifically **ChromaDB** and **FAISS**. The design includes an abstract base class (`VectorDB`) and concrete implementations, allowing easy integration and switching between different vector database backends.

---

## 🚀 Key Features

* **Unified Interface**: A single abstract class to interact seamlessly with multiple vector database backends.
* **Plug-and-Play**: Quickly switch between **ChromaDB** and **FAISS** implementations with no changes to your codebase.
* **Automatic Persistence**: Data persistence built-in—automatic saving/loading of embeddings and associated metadata.
* **Sentence Embedding Integration**: Uses the powerful `sentence-transformers` library for generating high-quality embeddings.
* **Easy-to-Use Factory Pattern**: Instantiate your desired vector database easily through a simple factory method.

---

## 📁 Project Structure

```
project-root/
├── db/                       # Persistent storage for ChromaDB and FAISS
├── src/                      # Source code
│   └── vectordb.py           # Abstract base class and implementations
├── requirements.txt          # Python dependencies
└── run_example.py            # Demonstration script for quick testing
```

---

## ⚙️ Installation

Clone this repository and install the dependencies:

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
pip install -r requirements.txt
```

---

## 🚦 Quick Start

### Run the Example

To quickly test the setup, run:

```bash
python run_example.py
```

This script demonstrates adding texts and performing similarity searches using both ChromaDB and FAISS.

---

## 🧑‍💻 Usage Example

```python
from src.vectordb import VectorDBFactory

# Create a ChromaDB instance
vectordb = VectorDBFactory.create_vectordb('chroma', db_path='db/chroma', collection_name='my_collection')

# Add texts with metadata
texts = ["Hello World", "Machine learning is fun!", "Vector databases rock!"]
metadata = [{"source": "example"} for _ in texts]
vectordb.add_texts(texts, metadata)

# Perform a search
results = vectordb.search("machine learning", k=2)
print(results)
```

Switch to FAISS by simply changing the factory call:

```python
vectordb = VectorDBFactory.create_vectordb('faiss', db_path='db/faiss')
```

---

## 📌 Dependencies

* [ChromaDB](https://github.com/chroma-core/chroma)
* [FAISS](https://github.com/facebookresearch/faiss)
* [SentenceTransformers](https://github.com/UKPLab/sentence-transformers)
* [NumPy](https://github.com/numpy/numpy)

See [requirements.txt](requirements.txt) for full details.

---

## 🛠️ Contributing

Feel free to fork this repository, submit issues, or open pull requests. Contributions are welcome!


