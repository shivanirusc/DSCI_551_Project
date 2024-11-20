import pandas as pd
import sqlite3
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import streamlit as st

# Function to download NLTK resources
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

# Function to clean and tokenize input text
def clean_and_tokenize(input_text):
    # Convert to lowercase and tokenize
    input_text = input_text.lower()
    tokens = word_tokenize(input_text)
    return tokens

# Function to remove stopwords
def remove_stopwords(tokens):
    stop_words = set(stopwords.words("english"))
    return [token for token in tokens if token not in stop_words]

# Function to lemmatize tokens
def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

# Function to map tokens to column names dynamically
def map_tokens_to_columns(tokens, column_names):
    mapped_columns = []
    for token in tokens:
        for col in column_names:
            if token in col.lower():  # Compare in lowercase to avoid case sensitivity issues
                mapped_columns.append(col)
                break
    return mapped_columns

# Function to generate SQL query based on user input
def generate_sql_query(user_input, column_names, table_name):
    # Clean and tokenize the user input
    tokens = clean_and_tokenize(user_input)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)

    # Map the tokens to actual column names
    mapped_columns = map_tokens_to_columns(tokens, column_names)

    if not mapped_columns:
        return "No matching columns found in your input. Please try again.", None

    # Identify quantitative and categorical columns
    quantitative_columns = [col for col in mapped_columns if col not in ['category', 'material', 'color', 'location', 'season', 'store_type', 'brand']]
    categorical_columns = [col for col in mapped_columns if col in ['category', 'material', 'color', 'location', 'season', 'store_type', 'brand']]

    # Example query pattern: "total <A> by <B>"
    if "total" in tokens or "sum" in tokens:
        for quant in quantitative_columns:
            for cat in categorical_columns:
                if quant in tokens and cat in tokens:
                    sql_query = f"SELECT {cat}, SUM({quant}) as total_{quant} FROM {table_name} GROUP BY {cat}"
                    nat_lang_query = f"Total {quant} by {cat}"
                    return nat_lang_query, sql_query

    # If no specific match, return a generic query
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

# Handle file upload and column extraction
if file:
    filename = file.name
    if filename.endswith('.csv'):
        data = pd.read_csv(file)
        uploaded_columns = data.columns.tolist()
        table_name = filename[:-4]
        # Store data in SQLite (or MongoDB if needed)
        conn = sqlite3.connect("chatdb_sql.db")
        data.to_sql(table_name, conn, if_exists='replace', index=False)
        st.write(f"**Uploaded Successfully!** Columns in your data: {uploaded_columns}")
    elif filename.endswith('.json'):
        data = pd.read_json(file)
        uploaded_columns = data.columns.tolist()
        table_name = filename[:-5]
        # Store data in MongoDB (or SQLite if needed)
        st.write(f"**Uploaded Successfully!** Columns in your data: {uploaded_columns}")
    else:
        st.write("**Unsupported file type. Please upload a CSV or JSON file.**")

# Chat interface
st.write("---")
st.write("**Chat with ChatDB:**")
user_input = st.text_input("Type your query here:")

if user_input and uploaded_columns:
    nat_lang_query, sql_query = generate_sql_query(user_input, uploaded_columns, table_name)

    if sql_query:
        st.write(f"**Natural Language Query:** {nat_lang_query}")
        st.code(sql_query)

        # Execute the generated SQL query on the dataset (SQLite example)
        conn = sqlite3.connect("chatdb_sql.db")
        result = pd.read_sql_query(sql_query, conn)
        st.write(f"**Query Result:**")
        st.dataframe(result)
    else:
        st.write(nat_lang_query)
elif user_input:
    st.write("Please upload a dataset first.")
