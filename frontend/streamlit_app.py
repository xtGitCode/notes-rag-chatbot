import streamlit as st
import requests
import json

BACKEND_URL = "http://127.0.0.1:8000"

def get_notes():
    try:
        response = requests.get(f"{BACKEND_URL}/notes/notes")
        response.raise_for_status() 
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching notes: {e}")
        return []

def create_note(title, content):
    try:
        headers = {'Content-Type': 'application/json'}
        data = {'title': title, 'content': content}
        response = requests.post(f"{BACKEND_URL}/notes/notes", headers=headers, data=json.dumps(data))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error creating note: {e}")
        return None

def delete_note(note_id):
    try:
        response = requests.delete(f"{BACKEND_URL}/notes/notes/{note_id}")
        response.raise_for_status()
        return response.status_code == 204
    except requests.exceptions.RequestException as e:
        st.error(f"Error deleting note: {e}")
        return False

st.title("My Notes App")

notes = get_notes()
num_notes = len(notes)

if notes:
    st.write(f"Total Notes: {num_notes}")
    for note in notes:
        with st.expander(note["title"]):
            st.write(note["content"])
            if st.button(f"Delete {note['title']}", key=f"delete_{note['id']}"):
                if delete_note(note["id"]):
                    st.success(f"Note '{note['title']}' deleted.")
                    st.rerun() 

else:
    st.info("No notes found. Create one below.")

# Initialize session state for title and content
if "new_title" not in st.session_state:
    st.session_state.new_title = ""
if "new_content" not in st.session_state:
    st.session_state.new_content = ""

st.subheader("Create New Note")
# Use session state variables for input fields
new_title = st.text_input("Title", value=st.session_state.new_title, key="title_input")
new_content = st.text_area("Content", value=st.session_state.new_content, key="content_input")

if st.button("Save Note"):
    if new_title and new_content:
        new_note = create_note(new_title, new_content)
        if new_note:
            st.success(f"Note '{new_title}' created!")
            st.session_state.new_title = ""
            st.session_state.new_content = ""
            st.rerun()
        else:
            st.error("Failed to create note.")
    else:
        st.warning("Please enter both title and content.")

# Chatbot section (to be implemented later)
st.subheader("Chat with your notes")
# ...