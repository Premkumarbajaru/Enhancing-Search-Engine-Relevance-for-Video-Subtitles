from Data_Extractor import DataExtractor
from Data_Cleaner import DataCleaner
from Vectordb import SubtitleVectorDB

def data_preprocessing_pipeline():
    print("🚀 Starting the subtitle processing pipeline...")

    # Step 1: Extract subtitles from the database
    extractor = DataExtractor(
        db_path=r"E:\Project\Innomatics\Sub_Search\Dataset\eng_subtitles_database.db",
        output_parquet=r"E:\Project\Innomatics\Sub_Search\subtitles_full.parquet"
    )
    extractor.extract_subtitles()
    print("✅ Subtitle extraction completed.")

    # Step 2: Clean subtitles
    cleaner = DataCleaner(
        input_parquet=r"E:\Project\Innomatics\Sub_Search\subtitles_full.parquet",
        output_parquet=r"E:\Project\Innomatics\Sub_Search\cleaned_subtitles.parquet"
    )
    cleaner.clean_subtitles()
    print("✅ Subtitle cleaning completed.")

    # Step 3: Store cleaned subtitles in a vector database
    vector_db = SubtitleVectorDB(
        db_path="./chroma_db",
        parquet_file=r"E:\Project\Innomatics\Sub_Search\cleaned_subtitles.parquet"
    )
    vector_db.load_data()
    print("✅ Subtitle vector database creation completed.")

    print("🎯 Processing pipeline executed successfully!")

if __name__ == "__main__":
    data_preprocessing_pipeline()

