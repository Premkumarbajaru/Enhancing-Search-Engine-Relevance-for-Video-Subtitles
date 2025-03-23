import sqlite3
import zipfile
import io
import chardet
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

class DataExtractor:
    def __init__(self, db_path, output_parquet, chunk_size=500, overlap=50):
        self.db_path = db_path
        self.output_parquet = output_parquet
        self.chunk_size = chunk_size
        self.overlap = overlap

    def extract_full_subtitle(self, content):
        zip_bytes = io.BytesIO(content)
        subtitle_text = ""

        try:
            with zipfile.ZipFile(zip_bytes, "r") as zip_ref:
                for file in zip_ref.namelist():
                    with zip_ref.open(file) as subtitle_file:
                        chunk = subtitle_file.read(1048576)  
                        detected_encoding = chardet.detect(chunk)["encoding"]

                        subtitle_file.seek(0)
                        subtitle_text = subtitle_file.read().decode(detected_encoding, errors="ignore")
                        break  

        except zipfile.BadZipFile:
            subtitle_text = "[Invalid ZIP File]"
        except Exception as e:
            subtitle_text = f"[Error: {str(e)}]"

        return subtitle_text

    def extract_subtitles(self):
        conn = sqlite3.connect(self.db_path)
        writer = None
        last_num = 0

        try:
            while True:
                query = f"""
                    SELECT num, name, content FROM zipfiles 
                    WHERE num > {last_num}
                    ORDER BY num ASC
                    LIMIT {self.chunk_size + self.overlap}
                """
                df_chunk = pd.read_sql_query(query, conn)
                if df_chunk.empty:
                    break

                df_chunk["subtitles"] = df_chunk["content"].apply(self.extract_full_subtitle)
                df_chunk.drop(columns=["content"], inplace=True)

                table = pa.Table.from_pandas(df_chunk)

                if writer is None:
                    writer = pq.ParquetWriter(self.output_parquet, table.schema)
                writer.write_table(table)

                last_num = df_chunk["num"].max()
                print(f"✅ Processed up to num {last_num}.")

        finally:
            if writer:
                writer.close()
        conn.close()
