import streamlit as st
import pandas as pd
import sqlite3
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pymongo import MongoClient
import os

# Download NLTK resources
def download_nltk_resources():
    try:
        nltk.download('punkt')  # Required for tokenization
        nltk.download('stopwords')  # For stopword removal
        nltk.download('wordnet')  # For lemmatization
        st.write("NLTK resources downloaded successfully.")
    except Exception as e:
        st.write(f"Error downloading NLTK resources: {e}")

# Call the download function
download_nltk_resources()

# Define the basic tokenizer (splitting on spaces and lowering text)
def basic_tokenizer(text):
    return text.lower().split()

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

# NLP Processing Function using both basic tokenizer and NLTK
def process_input(user_input):
    # Step 1: Tokenize using basic_tokenizer
    tokens = basic_tokenizer(user_input)
    
    # Step 2: Remove stopwords using NLTK stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Step 3: Lemmatize using NLTK WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens

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

    # Explicitly handle "sales amount" and "product category"
    sales_column = "sales_amount"  # Column name for sales amount
    category_column = "product_category"  # Column name for product category

    # Check if "total" or "sum" is mentioned in the processed tokens
    if "total" in processed_tokens or "sum" in processed_tokens:
        if sales_column in column_names and category_column in column_names:
            sql_query = f"SELECT {category_column}, SUM({sales_column}) as total_sales_amount FROM {table_name} GROUP BY {category_column}"
            nat_lang_query = f"Total sales amount by product category"
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
