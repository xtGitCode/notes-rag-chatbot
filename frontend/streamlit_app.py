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

def query_note(question):
    with st.spinner("Querying the chatbot..."):
        try:
            headers = {'Content-Type': 'application/json'}
            data = {'question': question}
            response = requests.post(f"{BACKEND_URL}/chat/chat", headers=headers, data=json.dumps(data))
            response.raise_for_status()
            response_data = response.json()

            if response_data:
                st.write(f"**Answer:** {response_data['answer']}")

                # Title and Content Formatting (Collapsible with Toggle)
                if response_data.get('title') or response_data.get('content'):
                    with st.expander("View Source (Note Title & Content)"):
                        if response_data.get('title'):
                            st.write(f"**Title:** {response_data['title']}")
                        if response_data.get('content'):
                            st.write(f"**Content:**")
                            st.markdown(response_data['content'], unsafe_allow_html=False)  # Prevent XSS

                else:
                    st.info("No notes found matching your question.")

            return response_data

        except requests.exceptions.RequestException as e:
            st.error(f"Error querying note: {e}")
            return False

st.title("Notes Chat")

st.subheader("My Notes")
notes = get_notes()
num_notes = len(notes)
st.write(f"Total Notes: {num_notes}")

if notes:
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
if "new_query" not in st.session_state:
    st.session_state.new_query = ""

st.subheader("Create New Note")
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

# Chatbot section 
st.subheader("Chat with your notes")
new_query= st.text_area("Ask question about your notes", value=st.session_state.new_content, key="query_input")
if st.button("Ask Chatbot"):
  if new_query:
    if query_note(new_query):
      st.session_state.new_query = ""  # Clear query input after successful response
    else:
      st.error("Failed to query notes")
  else:
    st.warning("Please enter a question")
