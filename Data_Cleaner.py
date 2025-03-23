import pyarrow.parquet as pq
import pyarrow as pa
import re

class DataCleaner:
    def __init__(self, input_parquet, output_parquet):
        self.input_parquet = input_parquet
        self.output_parquet = output_parquet

    def clean_text(self, text):
        unwanted_phrases = [
            r'(?i)api\.OpenSubtitles\.org is deprecated.*?\n?',
            r'(?i)implement REST API from OpenSubtitles\.com.*?\n?',
            r'(?i)ENJOY ALL VOD IN HIGH QUALITY.*?\n?',
            r'(?i)GET LIVE TV,MOVIES,SHOWS.*?\n?',
            r'(?i)Support us and become VIP.*?\n?',
            r'(?i)to remove all ads from.*?\n?',
            r'(?i)Watch any video online with Open-SUBTITLES.*?\n?',
            r'(?i)Free Browser extension: osdb\.link/ext.*?\n?',
            r'(?i)~ subtitles started by .*? ~\n?',
            r'(?i)~ edits & sync by .*? ~\n?',
            r'(?i)Advertise your product or brand here.*?\n?',
            r'(?i)contact www\.OpenSubtitles\.org.*?\n?',
            r'(?i)ENJOY ALL VOD IN HIGH QUALITY @ KVOD.TV.*?\n?',
            r'(?i)member.*?\n?',
            r'(?i)www\.OpenSubtitles\.org.*?\n?',
            r'(?i)Use the free code JOINNOW at.*?\n?',
            r'(?i)www\.playships\.eu.*?\n?',
            r'(?i)-== \[ www\.OpenSubtitles\.com \] ==-.*?\n?',
            r'(?i)please rate this subtitle at www\.osdb\.link/[\w\d]+\n?',
            r'(?i)help other users to choose the best subtitles\n?'
        ]

        # Remove unwanted phrases
        for phrase in unwanted_phrases:
            text = re.sub(phrase, '', text, flags=re.IGNORECASE)

        if "[Script Info]" in text or "[V4+ Styles]" in text:
            text = re.sub(r'\[.*?\](?:\n|$)', '', text, flags=re.MULTILINE)
            text = re.sub(r'\{\\.*?\}', '', text)
            text = re.sub(r'\{[^}]*\}', '', text)
            text = re.sub(r'[^\x00-\x7F]+', '', text)
            dialogues = re.findall(r'^Dialogue: \d+,\d{1,2}:\d{2}:\d{2}\.\d{2},\d{1,2}:\d{2}:\d{2}\.\d{2},[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,(.*)', text, flags=re.MULTILINE)
        else:
            text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
            text = re.sub(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', '', text)
            text = re.sub(r'<[^>]+>', '', text)
            dialogues = text.split("\n")

        # Clean and convert to lowercase
        dialogues = [line.strip().lower() for line in dialogues if line.strip()]
        
        return "\n".join(dialogues)

    def clean_subtitles(self):
        parquet_file = pq.ParquetFile(self.input_parquet)
        writer = None  

        try:
            for i, batch in enumerate(parquet_file.iter_batches(batch_size=10_000)):
                df_chunk = batch.to_pandas()

                if 'subtitles' not in df_chunk.columns:
                    raise KeyError("The 'subtitles' column is missing in the Parquet file.")

                df_chunk['subtitles'] = df_chunk['subtitles'].astype(str).apply(self.clean_text)

                table = pa.Table.from_pandas(df_chunk)

                if writer is None:
                    writer = pq.ParquetWriter(self.output_parquet, table.schema)
                writer.write_table(table)

                print(f"✅ Processed batch {i+1} with {len(df_chunk)} rows")

        finally:
            if writer:
                writer.close()

        print(f"✅ Cleaned subtitles saved to: {self.output_parquet}")