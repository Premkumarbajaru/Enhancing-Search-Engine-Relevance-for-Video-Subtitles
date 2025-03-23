import chromadb
import dask.dataframe as dd
import torch
from sentence_transformers import SentenceTransformer

class SubtitleVectorDB:
    def __init__(self, db_path, parquet_file, model_name="all-MiniLM-L6-v2", batch_size=1000, overlap=100):
        self.db_path = db_path
        self.parquet_file = parquet_file
        self.batch_size = batch_size
        self.overlap = overlap  # Overlapping chunk size
        
        # Load embedding model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.chroma_client.get_or_create_collection(name="subtitles_collection")
    
    def load_data(self):
        """Loads and processes the Parquet file into ChromaDB with overlapping chunks."""
        # Read Parquet file using Dask
        dask_df = dd.read_parquet(self.parquet_file, blocksize="500MB")
        pandas_df = dask_df.compute()

        # Process data with overlap
        batches = []
        num_rows = len(pandas_df)
        for i in range(0, num_rows, self.batch_size):
            start_idx = max(0, i - self.overlap)
            end_idx = min(num_rows, i + self.batch_size)
            batch = pandas_df.iloc[start_idx:end_idx]
            batches.append(batch)

        for batch_count, batch_df in enumerate(batches):
            batch_data = {
                "ids": batch_df["num"].astype(str).tolist(),
                "documents": batch_df["subtitles"].tolist(),
                "metadatas": [{"name": name} for name in batch_df["name"]],
                "embeddings": self.model.encode(batch_df["subtitles"].tolist(), convert_to_numpy=True).tolist()
            }
            
            self.collection.add(**batch_data)
            print(f"✅ Processed batch {batch_count + 1} with {len(batch_df)} records (including overlap).")
        
        print("🚀 ChromaDB vector database created successfully!")

if __name__ == "__main__":
    db_loader = SubtitleVectorDB(db_path="./chroma_db", parquet_file=r"E:\Project\Innomatics\Sub_Search\cleaned_subtitles.parquet")
    db_loader.load_data()
