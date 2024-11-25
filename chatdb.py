# import streamlit as st
# import pandas as pd
# import sqlite3
# from pymongo import MongoClient
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# import os

# from sql_sample_queries import categorize_columns, generate_sample_queries, generate_construct_queries

# # Download NLTK resources
# def download_nltk_resources():
#     try:
#         nltk.download('punkt')  # Required for tokenization
#         nltk.download('stopwords')  # For stopword removal
#         nltk.download('wordnet')  # For lemmatization
#         st.write("NLTK resources downloaded successfully.")
#     except Exception as e:
#         st.write(f"Error downloading NLTK resources: {e}")

# # Call the download function
# download_nltk_resources()

# # Define the basic tokenizer (splitting on spaces and lowering text)
# def basic_tokenizer(text):
#     return text.lower().split()

# # Initialize MongoDB
# mongo_client = MongoClient("mongodb://localhost:27017/")
# mongo_db = mongo_client["chatdb"]
# sqldb_list = []

# # Helper function to check allowed file types
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'json'}

# # Function to store data in MongoDB
# def store_in_mongodb(data, json_name):
#     collection_name = json_name[:-5]  # Remove '.json' from filename
#     collection = mongo_db[collection_name]
#     collection.drop()  # Clear old data before inserting new
#     collection.insert_many(data.to_dict(orient='records'))
#     return data.columns.tolist()

# # Function to store data in SQLite
# def make_sql_db(df, csv_name):
#     conn = sqlite3.connect("chatdb_sql.db")  # SQLite database file
#     table_name = csv_name[:-4]  # Remove '.csv' from filename
#     df.to_sql(table_name, conn, if_exists='replace', index=False)
#     sqldb_list.append(table_name)
#     return df.columns.tolist()

# # NLP Processing Function using both basic tokenizer and NLTK
# def process_input(user_input):
#     # Step 1: Tokenize using basic_tokenizer
#     tokens = basic_tokenizer(user_input)
    
#     # Step 2: Remove stopwords using NLTK stopwords
#     stop_words = set(stopwords.words("english"))
#     tokens = [token for token in tokens if token not in stop_words]
    
#     # Step 3: Lemmatize using NLTK WordNetLemmatizer
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
#     return tokens

# # Function to generate SQL queries based on user input
# def generate_sql_query(user_input, column_names, table_name):
#     # Clean and tokenize the user input
#     tokens = process_input(user_input)
    
#     # Map the tokens to actual column names
#     mapped_columns = [col for col in column_names if col.lower() in tokens]

#     if not mapped_columns:
#         return "No matching columns found in your input. Please try again.", None

#     # Identify quantitative and categorical columns
#     quantitative_columns = [col for col in mapped_columns if col not in ['category', 'material', 'color', 'location', 'season', 'store_type', 'brand']]
#     categorical_columns = [col for col in mapped_columns if col in ['category', 'material', 'color', 'location', 'season', 'store_type', 'brand']]

#     # Example query pattern: "total <A> by <B>"
#     if "total" in tokens or "sum" in tokens:
#         for quant in quantitative_columns:
#             for cat in categorical_columns:
#                 if quant in tokens and cat in tokens:
#                     sql_query = f"SELECT {cat}, SUM({quant}) as total_{quant} FROM {table_name} GROUP BY {cat}"
#                     nat_lang_query = f"Total {quant} by {cat}"
#                     return nat_lang_query, sql_query

#     # Example pattern: "average <A> by <B>"
#     if "average" in tokens or "avg" in tokens:
#         for quant in quantitative_columns:
#             for cat in categorical_columns:
#                 if quant in tokens and cat in tokens:
#                     sql_query = f"SELECT {cat}, AVG({quant}) as average_{quant} FROM {table_name} GROUP BY {cat}"
#                     nat_lang_query = f"Average {quant} by {cat}"
#                     return nat_lang_query, sql_query

#     # Example pattern: "maximum <A> by <B>"
#     if "maximum" in tokens or "max" in tokens:
#         for quant in quantitative_columns:
#             for cat in categorical_columns:
#                 if quant in tokens and cat in tokens:
#                     sql_query = f"SELECT {cat}, MAX({quant}) as max_{quant} FROM {table_name} GROUP BY {cat}"
#                     nat_lang_query = f"Maximum {quant} by {cat}"
#                     return nat_lang_query, sql_query

#     # Example pattern: "count of <A> by <B>"
#     if "count" in tokens or "total" in tokens:
#         for cat in categorical_columns:
#             if cat in tokens:
#                 sql_query = f"SELECT {cat}, COUNT(*) as count_{cat} FROM {table_name} GROUP BY {cat}"
#                 nat_lang_query = f"Count of {cat}"
#                 return nat_lang_query, sql_query

#     # Example pattern: "total <A> where <B>"
#     if "where" in tokens:
#         for quant in quantitative_columns:
#             if quant in tokens:
#                 condition = ' '.join(tokens[tokens.index("where")+1:])
#                 sql_query = f"SELECT SUM({quant}) as total_{quant} FROM {table_name} WHERE {condition}"
#                 nat_lang_query = f"Total {quant} where {condition}"
#                 return nat_lang_query, sql_query

#     # Example pattern: "top N <A> by <B>"
#     if "top" in tokens and "by" in tokens:
#         for quant in quantitative_columns:
#             for cat in categorical_columns:
#                 if quant in tokens and cat in tokens:
#                     n_value = 5  # Default top 5, could be extracted from input if specified
#                     sql_query = f"SELECT {cat}, SUM({quant}) as total_{quant} FROM {table_name} GROUP BY {cat} ORDER BY total_{quant} DESC LIMIT {n_value}"
#                     nat_lang_query = f"Top {n_value} {cat} by {quant}"
#                     return nat_lang_query, sql_query

#     # If no specific match, return a generic query
#     return "Query could not be interpreted. Please try rephrasing.", None

# # Streamlit app setup
# st.title("ChatDB: Interactive Query Assistant")
# st.write(
#     "Upload your dataset (CSV or JSON), and ask ChatDB to generate SQL queries for your data using natural language."
# )

# # File upload section
# file = st.file_uploader("Choose a CSV or JSON file to populate your database:", type=["csv", "json"])
# uploaded_columns = []
# table_name = ""

# if file:
#     filename = file.name
#     if allowed_file(filename):
#         if filename.endswith('.csv'):
#             data = pd.read_csv(file)
#             uploaded_columns = make_sql_db(data, filename)
#             table_name = filename[:-4]
#         else:
#             data = pd.read_json(file)
#             uploaded_columns = store_in_mongodb(data, filename)
#             table_name = filename[:-5]

#         st.write(f"**Uploaded Successfully!** Columns in your data: {uploaded_columns}")
#     else:
#         st.write("**Unsupported file type. Please upload a CSV or JSON file.**")

# # Chat interface
# st.write("---")
# st.write("**Chat with ChatDB:**")
# user_input = st.text_input("Type your query here:")

# if user_input and uploaded_columns:
#     # Handle "example sql query"
#     if user_input.lower() == "example sql query":
#         if table_name:  # Ensure a table is available
#             # Categorize columns
#             categorical, quantitative = categorize_columns(data)
#             if categorical and quantitative:
#                 # Generate sample queries
#                 sample_queries = generate_sample_queries(table_name, categorical, quantitative)

#                 # Format the output
#                 st.write("Here are some example SQL queries:")
#                 for sample_query in sample_queries:
#                     st.code(sample_query)
#             else:
#                 st.write("Your dataset does not have the necessary columns for sample SQL queries.")
#         else:
#             st.write("Please upload a dataset first to generate example queries.")
        
            
#     else:
#         nat_lang_query, sql_query = generate_sql_query(user_input, uploaded_columns, table_name)

#         if sql_query:
#             st.write(f"**Natural Language Query:** {nat_lang_query}")
#             st.code(sql_query)

#             # Execute the query on SQLite database
#             if filename.endswith('.csv'):
#                 conn = sqlite3.connect("chatdb_sql.db")
#                 result = pd.read_sql_query(sql_query, conn)
#                 st.write("**Query Result from SQLite:**")
#                 st.dataframe(result)

#             # Execute the query on MongoDB database
#             elif filename.endswith('.json'):
#                 collection = mongo_db[table_name]
#                 pipeline = [
#                     {"$group": {"_id": f"${processed_tokens[1]}", f"total_{processed_tokens[3]}": {"$sum": f"${processed_tokens[3]}"}}}
#                 ]
#                 result = list(collection.aggregate(pipeline))
#                 result_df = pd.DataFrame(result)
#                 st.write("**Query Result from MongoDB:**")
#                 st.dataframe(result_df)

#         else:
#             st.write(nat_lang_query)
# elif user_input:
#     st.write("Please upload a dataset first.")

# # Display chat history
# if 'chat_history' not in st.session_state:
#     st.session_state['chat_history'] = []

# if user_input:
#     if user_input.lower() == "example sql query":
#         st.session_state['chat_history'].append({"user": user_input, "response": sample_query})
#     else:
#         st.session_state['chat_history'].append({"user": user_input, "response": nat_lang_query if sql_query else "Unable to generate query."})

# for chat in st.session_state['chat_history']:
#     st.write(f"**You:** {chat['user']}")
#     st.write(f"**ChatDB:** {chat['response']}")
import streamlit as st
import pandas as pd
import sqlite3
from pymongo import MongoClient
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import re

from sql_sample_queries import generate_sample_queries

# Download NLTK resources
def download_nltk_resources():
    try:
        nltk.download('punkt')  # Required for tokenization
        nltk.download('stopwords')  # For stopword removal
        nltk.download('wordnet')  # For lemmatization
    except Exception as e:
        st.write(f"Error downloading NLTK resources: {e}")

# Call the download function
download_nltk_resources()

# Function to tokenize input
def basic_tokenizer(user_input):
    # Remove punctuation and tokenize
    tokens = re.sub(r'[^\w\s]', '', user_input.lower()).split()
    return tokens

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

# Function to map the tokens dynamically to column names in the dataset
def map_columns(tokens, columns):
    for token in tokens:
        for column in columns:
            if token.lower() in column.lower():
                return column
    return None

# Example function for categorizing columns
def categorize_columns(dataframe):
    categorical = [col for col in dataframe.columns if dataframe[col].dtype == 'object']
    quantitative = [col for col in dataframe.columns if dataframe[col].dtype in ['int64', 'float64']]
    return categorical, quantitative

def generate_sql_query(user_input, uploaded_columns, table_name, data):
    # Step 1: Process the input query using NLP processing (basic tokenization and stemming)
    tokens = process_input(user_input)
    
    st.write(f"Tokens extracted: {tokens}")

    # Step 2: Categorize columns in the DataFrame (categorical vs. quantitative)
    categorical_columns, quantitative_columns = categorize_columns(data)
    st.write(f"categorical_columns extracted: {categorical_columns}")
    st.write(f"quantitative_columns extracted: {quantitative_columns}")
    
    # Handle sum and total queries
    if "sum" in tokens or "total" in tokens:
        column = map_columns(tokens, quantitative_columns)  # Identify the quantitative column
        group_by_column = map_columns(tokens, categorical_columns)  # Identify the categorical column

        if column and group_by_column:  # Ensure both column mappings exist
            sql_query = f"SELECT {group_by_column}, SUM({column}) as total_{column} FROM {table_name} GROUP BY {group_by_column}"
            nat_lang_query = f"Sum of {column} grouped by {group_by_column}"
            return nat_lang_query, sql_query
    
    # Handle 'count' queries
    if "count" in tokens or "many" in tokens:  # Include synonyms like "many"
     for cat in categorical_columns:
         # Check for exact or partial matches
         if any(token in cat.lower() for token in tokens):
             sql_query = f"SELECT {cat}, COUNT(*) as count_{cat} FROM {table_name} GROUP BY {cat}"
             nat_lang_query = f"Count of {cat}"
             return nat_lang_query, sql_query
    
    # Handle 'average' or 'avg' queries
    if any(word in tokens for word in ["average", "avg"]):
        quant_col = None
        cat_col = None
    
        # Match quantitative and categorical columns
        for quant in quantitative_columns:
            if any(token in quant.lower() for token in tokens):
                quant_col = quant
                break  # Exit loop once a match is found
        
        for cat in categorical_columns:
            if any(token in cat.lower() for token in tokens):
                cat_col = cat
                break  # Exit loop once a match is found
        # Generate SQL query if both columns are matched
        if quant_col and cat_col:
            sql_query = f"SELECT {cat_col}, AVG({quant_col}) as average_{quant_col} FROM {table_name} GROUP BY {cat_col}"
            nat_lang_query = f"Average {quant_col} by {cat_col}"
            return nat_lang_query, sql_query

    # Step 6: Handle 'maximum' or 'max' queries
    if any(word in tokens for word in ["maximum", "max"]):
        # Match quantitative column
        matched_column = None
        for quant in quantitative_columns:
            if any(token in quant.lower() for token in tokens):
                matched_column = quant
                break  # Exit loop once a match is found

        # Generate SQL query if a quantitative column is matched
        if matched_column:
            sql_query = f"SELECT '{matched_column}', MAX({matched_column}) as max_{matched_column} FROM {table_name}"
            nat_lang_query = f"Maximum {matched_column}"
            return nat_lang_query, sql_query

    # Step 7: Handle 'minimum' or 'min' queries
    if any(word in tokens for word in ["minimum", "min"]):
        # Match quantitative column
        matched_column = None
        for quant in quantitative_columns:
            if any(token in quant.lower() for token in tokens):
                matched_column = quant
                break  # Exit loop once a match is found
        # Generate SQL query if a quantitative column is matched
        if matched_column:
            sql_query = f"SELECT '{matched_column}', MIN({matched_column}) as min_{matched_column} FROM {table_name}"
            nat_lang_query = f"Minimum {matched_column}"
            return nat_lang_query, sql_query

    # Step 8: Handle comparison queries ('less than', 'greater than', 'equal to')
    if any(word in tokens for word in ["less", "greater", "equal", "not"]):
        # Identify the quantitative column and value for comparison
        matched_column = None
        comparison_operator = None
        comparison_value = None

        # Match a quantitative column
        for quant in quantitative_columns:
            if any(token in quant.lower() for token in tokens):
                matched_column = quant
                break  # Exit loop once a match is found

        # Identify comparison operator
        if "less" in tokens:
            comparison_operator = "<"
        elif "greater" in tokens:
            comparison_operator = ">"
        elif "equal" in tokens:
            comparison_operator = "="
        elif "not" in tokens and "equal" in tokens:
            comparison_operator = "!="

        # Extract the value for comparison
        for token in normalized_tokens:
            if token.isdigit():
                comparison_value = token
                break  # Exit loop once a number is found

        # Generate SQL query if both column and value are identified
        if matched_column and comparison_operator and comparison_value:
            sql_query = f"SELECT * FROM {table_name} WHERE {matched_column} {comparison_operator} {comparison_value}"
            nat_lang_query = f"Rows where {matched_column} is {comparison_operator} {comparison_value}"
            return nat_lang_query, sql_query
    
    # Step 7: Handle conditional queries (e.g., where quantity > 100)
    if "where" in tokens:
        condition_index = tokens.index("where") + 1
        condition = ' '.join(tokens[condition_index:])
        for quant in quantitative_columns:
            if quant in tokens:
                sql_query = f"SELECT {quant} FROM {table_name} WHERE {condition}"
                nat_lang_query = f"{quant} where {condition}"
                print(f"Generated query: {sql_query}")
                return nat_lang_query, sql_query

    # Fallback in case no match is found
    return "Query could not be interpreted. Please try rephrasing.", ""

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
        if table_name:  # Ensure a table is available
            # Categorize columns
            categorical, quantitative = categorize_columns(data)
            if categorical and quantitative:
                # Generate sample queries
                sample_queries = generate_sample_queries(table_name, categorical, quantitative)

                # Format the output
                st.write("Here are some example SQL queries:")
                for sample_query in sample_queries:
                    st.code(sample_query)
            else:
                st.write("Your dataset does not have the necessary columns for sample SQL queries.")
        else:
            st.write("Please upload a dataset first to generate example queries.")
        
            
    else:
        nat_lang_query, sql_query = generate_sql_query(user_input, uploaded_columns, table_name, data)

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
    else:
        st.session_state['chat_history'].append({"user": user_input, "response": nat_lang_query if sql_query else "Unable to generate query."})

for chat in st.session_state['chat_history']:
    st.write(f"**You:** {chat['user']}")
    st.write(f"**ChatDB:** {chat['response']}")
