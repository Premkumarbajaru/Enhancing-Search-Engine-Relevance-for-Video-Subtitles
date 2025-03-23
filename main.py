import streamlit as st
import os
import sqlite3
import uuid
import json
from dotenv import load_dotenv
from audio_handler import AudioProcessor
from query_extraction import SubtitleVectorDB
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()
API_KEY = os.getenv("API_KEY")

DB_PATH = "./chroma_db"
HISTORY_DB = "./history.db"
USER_JSON = "./users.json"

if not os.path.exists(USER_JSON):
    with open(USER_JSON, "w") as f:
        json.dump([], f)

def load_users():
    with open(USER_JSON, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USER_JSON, "w") as f:
        json.dump(users, f, indent=4)

def get_session_id(username):
    users = load_users()
    for user in users:
        if user["name"] == username:
            return user["uuid"]
    return None

def add_new_user(username):
    users = load_users()
    new_uuid = str(uuid.uuid4())
    users.append({"name": username, "uuid": new_uuid})
    save_users(users)
    return new_uuid

def initialize_db():
    conn = sqlite3.connect(HISTORY_DB)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history (
                 session_id TEXT, query TEXT, response TEXT)''')
    conn.commit()
    conn.close()

initialize_db()

def load_history(session_id):
    conn = sqlite3.connect(HISTORY_DB)
    c = conn.cursor()
    c.execute("SELECT query, response FROM chat_history WHERE session_id = ?", (session_id,))
    history = c.fetchall()
    conn.close()
    return history

def save_history(session_id, query, response):
    conn = sqlite3.connect(HISTORY_DB)
    c = conn.cursor()
    c.execute("INSERT INTO chat_history (session_id, query, response) VALUES (?, ?, ?)",
              (session_id, query, response))
    conn.commit()
    conn.close()

db = SubtitleVectorDB(db_path=DB_PATH)

def setup_chat_model():
    return ChatGoogleGenerativeAI(api_key=API_KEY, model="gemini-1.5-pro", temperature=0.7)

def create_prompt(query, related_data):
    filtered_movies = [(movie, score) for movie, score in related_data if 0 <= score <= 1]
    movie_list = "\n".join([
        f"- {movie} (Relevance Score: {score:.2f})"
        for movie, score in filtered_movies
    ]) if filtered_movies else "No relevant movies found."

    explanation_prompt = (
        "Respond only based on the retrieved movies and their relevance scores. "
        "Do not make assumptions or suggest movies outside the list."
    ) if filtered_movies else "Inform the user that no relevant movies were retrieved."

    return ChatPromptTemplate.from_messages([
        SystemMessage(content=f"""
        You are an AI-powered movie search assistant.

        **User Query:** {query}
        
        **Relevant Movies Retrieved:**
        {movie_list}
        
        {explanation_prompt}
        """),
        MessagesPlaceholder(variable_name="history"),
        HumanMessage(content="{human_input}")
    ])

def generate_response(user_query):
    related_data = db.query_subtitles(user_query, top_k=3)
    chat_model = setup_chat_model()
    chat_prompt = create_prompt(user_query, related_data)

    conversation_chain = RunnableWithMessageHistory(
        chat_prompt | chat_model | StrOutputParser(),
        lambda session_id: st.session_state.chat_history,
        input_messages_key="human_input",
        history_messages_key="history"
    )

    response = conversation_chain.invoke(
        {"human_input": user_query},
        {"configurable": {"session_id": st.session_state.session_id}}
    )

    if response:
        st.session_state.chat_history.add_message(HumanMessage(content=user_query))
        st.session_state.chat_history.add_message(SystemMessage(content=response))
        save_history(st.session_state.session_id, user_query, response)

    return response

def process_chat_input(user_input):
    if user_input:
        if not st.session_state.username:
            st.warning("Please select or create a user session first.")
            return  

        ai_response = generate_response(user_input)

        if not st.session_state.display_history or (st.session_state.display_history[-1] != (user_input, ai_response)):
            st.session_state.display_history.append((user_input, ai_response))

        st.rerun() 

def main():
    st.set_page_config(page_title="Sub Search Chatbot", layout="wide")
    st.title("🎬 Sub Search Chatbot")

    if "username" not in st.session_state:
        st.session_state.username = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "display_history" not in st.session_state:
        st.session_state.display_history = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ChatMessageHistory()

    st.sidebar.header("User Session")
    users = load_users()
    user_options = [user["name"] for user in users] + ["Create New User"]
    selected_user = st.sidebar.selectbox("Select or create a new user:", user_options, key="selected_user")

    if selected_user == "Create New User":
        new_user = st.sidebar.text_input("Enter new user name:", key="new_user_input")
        if st.sidebar.button("Submit New User", key="submit_new_user"):
            if new_user.strip():
                st.session_state.username = new_user.strip()
                st.session_state.session_id = add_new_user(st.session_state.username)
                st.rerun()

    elif selected_user:
        if st.sidebar.button("Load Session", key="load_session"):
            if st.session_state.username != selected_user:
                st.session_state.username = selected_user
                st.session_state.session_id = get_session_id(selected_user)
                st.session_state.display_history = load_history(st.session_state.session_id)
                st.session_state.chat_history = ChatMessageHistory()
                st.rerun()

    if st.session_state.username:
        st.sidebar.subheader(f"Active Session: {st.session_state.username}")

        input_mode = st.sidebar.radio("Choose Input Mode:", ["Text Chat", "Voice Input"])

        chat_container = st.container()
        with chat_container:
            for query, response in st.session_state.display_history:
                st.chat_message("user", avatar="👤").write(query)
                st.chat_message("ai", avatar="🤖").write(response)

        if input_mode == "Text Chat":
            user_input = st.text_input("Type your message:", key="user_input")
            if st.button("Send", key="send_button"):
                process_chat_input(user_input)

        elif input_mode == "Voice Input":
            recorded_audio = st.audio_input("Record your audio file")
            if recorded_audio is not None:
                audio_processor = AudioProcessor()
                query = audio_processor.transcribe_audio(audio_processor.save_audio(recorded_audio))
                if st.button("Send", key="send_button"):
                    process_chat_input(query)

if __name__ == "__main__":
    main()
