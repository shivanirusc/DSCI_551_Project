import streamlit as st
import pandas as pd
import sqlite3
from pymongo import MongoClient
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

from sql_queries import categorize_columns, generate_sample_queries, generate_construct_queries, execute_query

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
def generate_sql_query(user_input, column_names, table_name):
    # Clean and tokenize the user input
    tokens = process_input(user_input)
    
    # Map the tokens to actual column names
    mapped_columns = [col for col in column_names if col.lower() in tokens]

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
    # Handle "example sql query"
    if user_input.lower() == "example sql query":
        categorical, quantitative = categorize_columns(data)
        if categorical and quantitative:
            # Generate sample queries
            sample_queries = generate_sample_queries(table_name, categorical, quantitative)

            # Format the output
            st.write("Here are some example SQL queries:")
            for sample_query in sample_queries:
                # Print each query
                st.code(sample_query)
                # Executes query and shows result
                execute_query(sample_query)
        else:
            st.write("Your dataset does not have the necessary columns for sample SQL queries.")
        
    if "example query with" in user_input.lower():
        # Extract the construct from the user input
        construct = user_input.lower().replace("example query with", "").strip()

        # Categorize columns into categorical and quantitative
        categorical, quantitative = categorize_columns(data)

        if categorical and quantitative:
            # Generate queries based on the specified construct
            construct_queries = generate_construct_queries(construct, table_name, categorical, quantitative)

            if construct_queries:
                # Format the output
                st.write(f"Here are some example SQL queries using '{construct}':")
                for construct_query in construct_queries:
                    st.code(construct_query)
                    execute_query(construct_query)  # Assuming execute_query is a function that runs and displays the query
                    break
            else:
                st.write(f"No valid queries could be generated for the construct '{construct}'.")
        else:
            st.write("Your dataset does not have the necessary columns for construct-specific queries.")


    else:
        nat_lang_query, sql_query = generate_sql_query(user_input, uploaded_columns, table_name)

        if sql_query:
            st.write(f"**Natural Language Query:** {nat_lang_query}")
            st.code(sql_query)

            # Execute the query on SQLite database
            if filename.endswith('.csv'):
                execute_query(sql_query)

            # Execute the query on MongoDB database
            elif filename.endswith('.json'):
                collection = mongo_db[table_name]
                pipeline = [
                    {"$group": {"_id": f"${processed_tokens[1]}", f"total_{processed_tokens[3]}": {"$sum": f"${processed_tokens[3]}"}}}
                ]
                result = list(collection.aggregate(pipeline))
                result_df = pd.DataFrame(result)
                st.write("**Query Result from MongoDB:**")
                st.dataframe(result_df)

        else:
            st.write(nat_lang_query)
elif user_input:
    st.write("Please upload a dataset first.")

# Display chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if user_input:
    if user_input.lower() == "example sql query":
        st.session_state['chat_history'].append({"user": user_input, "response": sample_query})
    elif "example query with" in user_input.lower():
        st.session_state['chat_history'].append({"user": user_input, "response": construct_query})
    else:
        st.session_state['chat_history'].append({"user": user_input, "response": nat_lang_query if sql_query else "Unable to generate query."})

for chat in st.session_state['chat_history']:
    st.write(f"**You:** {chat['user']}")
    st.write(f"**ChatDB:** {chat['response']}")