import chromadb
from sentence_transformers import SentenceTransformer
import re

class SubtitleVectorDB:
    def __init__(self, db_path):
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(name="subtitles_collection")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def extract_movie_name(self, filename):
        """Extracts a clean movie name from the subtitle filename."""
        cleaned_name = re.sub(r"(\.s\d{2}|\.\d{3}|\(\d{4}\).*|\.eng.*)", "", filename, flags=re.IGNORECASE)
        return cleaned_name.replace(".", " ").strip().title()

    def query_subtitles(self, query, top_k=5):
        """Queries the vector database and returns movie names, subtitles, and similarity scores."""
        query_embedding = self.model.encode(query, convert_to_numpy=True).tolist()
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        
        output = []
        for i in range(len(results["documents"][0])):
            raw_movie_name = results["metadatas"][0][i]["name"]
            cleaned_movie_name = self.extract_movie_name(raw_movie_name)
            score = results["distances"][0][i]
            output.append((cleaned_movie_name, score))
        
        return output