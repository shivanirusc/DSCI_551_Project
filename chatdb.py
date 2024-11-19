import streamlit as st
import pandas as pd
import sqlite3
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pymongo import MongoClient
import os

# Debugging download process
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    st.write("NLTK resources downloaded successfully.")
except Exception as e:
    st.write(f"Error downloading NLTK resources: {e}")

# Initialize MongoDB
mongo_client = MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["chatdb"]
sqldb_list = []

# Helper function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'json'}

# Function to store data in MongoDB
def store_in_mongodb(data, json_name):
    collection_name = json_name[:-5]  # Remove '.json' from filename
    collection = mongo_db[collection_name]
    collection.drop()  # Clear old data before inserting new
    collection.insert_many(data.to_dict(orient='records'))
    return data.columns.tolist()

# Function to store data in SQLite
def make_sql_db(df, csv_name):
    conn = sqlite3.connect("chatdb_sql.db")  # SQLite database file
    table_name = csv_name[:-4]  # Remove '.csv' from filename
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    sqldb_list.append(table_name)
    return df.columns.tolist()

# NLP Processing Function
def process_input(user_input):
    # Tokenize the input after downloading NLTK resources
    tokens = word_tokenize(user_input.lower())
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    # Filter and lemmatize tokens
    return [
        lemmatizer.lemmatize(token)
        for token in tokens if token not in stop_words
    ]
# Function to generate SQL queries based on user input
def generate_sql_query(processed_tokens, column_names, table_name):
    quantitative = []
    categorical = []

    # Identify potential categorical and quantitative columns
    for col in column_names:
        if col.lower().endswith(('id', 'name', 'type', 'category', 'group')):
            categorical.append(col)
        else:
            quantitative.append(col)

    # Example query pattern: "total <A> by <B>"
    if "total" in processed_tokens or "sum" in processed_tokens:
        for quant in quantitative:
            for cat in categorical:
                if quant in processed_tokens and cat in processed_tokens:
                    sql_query = f"SELECT {cat}, SUM({quant}) as total_{quant} FROM {table_name} GROUP BY {cat}"
                    nat_lang_query = f"Total {quant} by {cat}"
                    return nat_lang_query, sql_query

    # If no specific match, provide a generic query
    return "Query could not be interpreted. Please try rephrasing.", None

# Streamlit app setup
st.title("ChatDB: Interactive Query Assistant")
st.write(
    "Upload your dataset (CSV or JSON), and ask ChatDB to generate SQL queries for your data using natural language."
)

# File upload section
file = st.file_uploader("Choose a CSV or JSON file to populate your database:", type=["csv", "json"])
uploaded_columns = []
table_name = ""

if file:
    filename = file.name
    if allowed_file(filename):
        if filename.endswith('.csv'):
            data = pd.read_csv(file)
            uploaded_columns = make_sql_db(data, filename)
            table_name = filename[:-4]
        else:
            data = pd.read_json(file)
            uploaded_columns = store_in_mongodb(data, filename)
            table_name = filename[:-5]

        st.write(f"**Uploaded Successfully!** Columns in your data: {uploaded_columns}")
    else:
        st.write("**Unsupported file type. Please upload a CSV or JSON file.**")

# Chat interface
st.write("---")
st.write("**Chat with ChatDB:**")
user_input = st.text_input("Type your query here:")

if user_input and uploaded_columns:
    processed_tokens = process_input(user_input)
    nat_lang_query, sql_query = generate_sql_query(processed_tokens, uploaded_columns, table_name)

    if sql_query:
        st.write(f"**Natural Language Query:** {nat_lang_query}")
        st.code(sql_query)
    else:
        st.write(nat_lang_query)
elif user_input:
    st.write("Please upload a dataset first.")

# Display chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if user_input:
    st.session_state['chat_history'].append({"user": user_input, "response": nat_lang_query if sql_query else "Unable to generate query."})

for chat in st.session_state['chat_history']:
    st.write(f"**You:** {chat['user']}")
    st.write(f"**ChatDB:** {chat['response']}")
