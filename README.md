# Project Readme

## Enhancing-Search-Engine-Relevance-for-Video-Subtitles

### Tech Stack
This project is a Sub Search Chatbot built with Streamlit and integrated with a subtitle vector database. It allows users to interact via text chat or voice input to query subtitles. The bot retrieves relevant movies based on the user query, processes the response using a Google Generative AI model, and stores the chat history in a SQLite database. It supports multiple user sessions, with each session linked to a unique user ID and stored in a JSON file. The system also allows users to load existing sessions or create new ones for continued interactions.

## Getting Started

Follow the steps below to set up and run the project:

### 1. Clone the Repository
```bash
git clone https://github.com/Premkumarbajaru/Enhancing-Search-Engine-Relevance-for-Video-Subtitles.git
cd Sub_Search
```

### 2. Save the Dataset
Place the `.db` file in the `Dataset` folder.

### 3. Run Data Preprocessing
Execute the `Data_Preprocessing.py` script to create the entire pipeline:
```bash
python Data_Preprocessing.py
```

### 4. Run the Main Script
Run the `main.py` file using Streamlit to start the project:
```bash
streamlit run main.py
```

Follow the instructions above to ensure proper setup and execution.
