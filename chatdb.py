# -*- coding: utf-8 -*-
"""chatDb.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1cFmOYrj0S0pMGvIcB_nAuQkuolNBlPTN
"""

pip install streamlit

import streamlit as st

# Title and description
st.title("ChatDB: Interactive Query Assistant")
st.write("Ask me anything about your database!")

# Chat History
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Input text box
user_input = st.text_input("Type your query here...")

if st.button("Submit"):
    # Append user input to chat history
    st.session_state['chat_history'].append({"user": user_input})

    # Placeholder: Add logic to handle user input and generate response
    response = "This is a placeholder response for your query."
    st.session_state['chat_history'].append({"bot": response})

# Display chat history
for chat in st.session_state['chat_history']:
    if "user" in chat:
        st.write(f"**You:** {chat['user']}")
    if "bot" in chat:
        st.write(f"**ChatDB:** {chat['bot']}")