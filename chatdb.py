import streamlit as st
import pandas as pd
import sqlite3
from pymongo import MongoClient
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
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

# Function to clean and tokenize the input text
def clean_and_tokenize(user_input):
    return word_tokenize(user_input)

# Function to remove stopwords from the tokenized input
def remove_stopwords(tokens):
    stop_words = set(stopwords.words("english"))
    return [token for token in tokens if token not in stop_words]

# Function to lemmatize the tokens
def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

# Function to map the tokens to the actual column names in the dataset
def map_tokens_to_columns(tokens, column_names):
    mapped_columns = []
    for token in tokens:
        for column in column_names:
            if token in column.lower():
                mapped_columns.append(column)
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

    # Example pattern: "average <A> by <B>"
    if "average" in tokens or "avg" in tokens:
        for quant in quantitative_columns:
            for cat in categorical_columns:
                if quant in tokens and cat in tokens:
                    sql_query = f"SELECT {cat}, AVG({quant}) as average_{quant} FROM {table_name} GROUP BY {cat}"
                    nat_lang_query = f"Average {quant} by {cat}"
                    return nat_lang_query, sql_query

    # Example pattern: "maximum <A> by <B>"
    if "maximum" in tokens or "max" in tokens:
        for quant in quantitative_columns:
            for cat in categorical_columns:
                if quant in tokens and cat in tokens:
                    sql_query = f"SELECT {cat}, MAX({quant}) as max_{quant} FROM {table_name} GROUP BY {cat}"
                    nat_lang_query = f"Maximum {quant} by {cat}"
                    return nat_lang_query, sql_query

    # Example pattern: "count of <A> by <B>"
    if "count" in tokens or "total" in tokens:
        for cat in categorical_columns:
            if cat in tokens:
                sql_query = f"SELECT {cat}, COUNT(*) as count_{cat} FROM {table_name} GROUP BY {cat}"
                nat_lang_query = f"Count of {cat}"
                return nat_lang_query, sql_query

    # Example pattern: "total <A> where <B>"
    if "where" in tokens:
        for quant in quantitative_columns:
            if quant in tokens:
                condition = ' '.join(tokens[tokens.index("where")+1:])
                sql_query = f"SELECT SUM({quant}) as total_{quant} FROM {table_name} WHERE {condition}"
                nat_lang_query = f"Total {quant} where {condition}"
                return nat_lang_query, sql_query

    # Example pattern: "top N <A> by <B>"
    if "top" in tokens and "by" in tokens:
        for quant in quantitative_columns:
            for cat in categorical_columns:
                if quant in tokens and cat in tokens:
                    n_value = 5  # Default top 5, could be extracted from input if specified
                    sql_query = f"SELECT {cat}, SUM({quant}) as total_{quant} FROM {table_name} GROUP BY {cat} ORDER BY total_{quant} DESC LIMIT {n_value}"
                    nat_lang_query = f"Top {n_value} {cat} by {quant}"
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

if file:
    filename = file.name
    if filename.endswith('.csv'):
        data = pd.read_csv(file)
        uploaded_columns = data.columns.tolist()
        table_name = filename[:-4]
        
        # Save to SQLite database
        conn = sqlite3.connect("chatdb_sql.db")
        data.to_sql(table_name, conn, if_exists='replace', index=False)
        st.write(f"**Uploaded Successfully!** Columns in your data: {uploaded_columns}")
    elif filename.endswith('.json'):
        data = pd.read_json(file)
        uploaded_columns = data.columns.tolist()
        table_name = filename[:-5]

        # Save to MongoDB database
        mongo_client = MongoClient("mongodb://localhost:27017/")
        mongo_db = mongo_client["chatdb"]
        collection = mongo_db[table_name]
        collection.drop()
        collection.insert_many(data.to_dict(orient='records'))
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

        # Execute the query on SQLite database
        if filename.endswith('.csv'):
            conn = sqlite3.connect("chatdb_sql.db")
            result = pd.read_sql_query(sql_query, conn)
            st.write("**Query Result from SQLite:**")
            st.dataframe(result)

        # Execute the query on MongoDB database
        if filename.endswith('.json'):
            collection = mongo_db[table_name]
            pipeline = [
                {"$group": {"_id": f"${uploaded_columns[1]}", f"total_{uploaded_columns[3]}": {"$sum": f"${uploaded_columns[3]}"}}}
            ]
            result = list(collection.aggregate(pipeline))
            result_df = pd.DataFrame(result)
            st.write("**Query Result from MongoDB:**")
            st.dataframe(result_df)
    else:
        st.write(nat_lang_query)
elif user_input:
    st.write("Please upload a dataset first.")
