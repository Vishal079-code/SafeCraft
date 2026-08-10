"""
Day 5b — Vector store: embed summary units and store/query them in ChromaDB.
Uses sentence-transformers (all-MiniLM-L6-v2) to generate embeddings,
stores them in a local, persistent ChromaDB collection.
"""

import json
import logging
from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer
from retriever.summary_builder import build_summary_units

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "repo_index"


class VectorStore:
    def __init__(self, db_path: str = CHROMA_DB_PATH, collection_name: str = COLLECTION_NAME):
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.client = chromadb.PersistentClient(path=db_path)
        # Explicitly specify cosine distance metric for standard similarity conversion
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def index_repo_map(self, repo_map: Dict[str, Any]):
        """Embed every function/class/method in repo_map and store them."""
        units = build_summary_units(repo_map)
        if not units:
            logger.warning("No summary units found - nothing to index.")
            return

        ids = [u["id"] for u in units]
        texts = [u["text"] for u in units]
        metadatas = [u["metadata"] for u in units]

        logger.info(f"Embedding {len(texts)} units with {EMBEDDING_MODEL_NAME}...")
        embeddings = self.model.encode(texts, show_progress_bar=False).tolist()

        # upsert so re-running after code changes updates existing entries
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info(f"Indexed {len(units)} units into ChromaDB collection '{self.collection.name}'")

    def query(self, prompt: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Given a natural language prompt, return the top_k most similar units."""
        if not prompt.strip():
            return []

        total_docs = self.collection.count()
        if total_docs == 0:
            logger.warning("Query attempted on an empty vector store.")
            return []

        # Clamp top_k to avoid requesting more results than exist in the index
        k = min(top_k, total_docs)

        query_embedding = self.model.encode([prompt]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k,
        )

        output = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return output

        for i in range(len(results["ids"][0])):
            dist = float(results["distances"][0][i])
            # For cosine distance, similarity = 1 - cosine_distance
            similarity_score = max(0.0, 1.0 - dist)

            output.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": dist,
                "vector_score": similarity_score,  # Standardized score key for hybrid ranker
                "score": similarity_score,
            })
        return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        with open("sample_repo/repo_map.json") as f:
            repo_map = json.load(f)

        store = VectorStore()
        store.index_repo_map(repo_map)

        print("\n--- Test query: 'how do I check a user's password' ---")
        results = store.query("how do I check a user's password", top_k=3)
        for r in results:
            print(f"  [{r['vector_score']:.3f}] {r['id']}  ->  {r['text']}")

        print("\n--- Test query: 'process a payment for a user' ---")
        results = store.query("process a payment for a user", top_k=3)
        for r in results:
            print(f"  [{r['vector_score']:.3f}] {r['id']}  ->  {r['text']}")
    except FileNotFoundError:
        print("Error: sample_repo/repo_map.json not found.")