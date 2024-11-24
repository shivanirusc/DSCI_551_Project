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

from sql_sample_queries import generate_sample_queries

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

# Example function for categorizing columns
def categorize_columns(dataframe):
    categorical = [col for col in dataframe.columns if dataframe[col].dtype == 'object']
    quantitative = [col for col in dataframe.columns if dataframe[col].dtype in ['int64', 'float64']]
    return categorical, quantitative

# Function to generate SQL queries based on user input
# def generate_sql_query(user_input, column_names, table_name, dataframe):
#     # Clean and tokenize the user input
#     tokens = process_input(user_input)
    
#     # Map the tokens to actual column names
#     mapped_columns = [col for col in column_names if col.lower() in tokens]

#     if not mapped_columns:
#         return "No matching columns found in your input. Please try again.", None

#     # Categorize columns based on the dataframe
#     categorical_columns, quantitative_columns = categorize_columns(dataframe)

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
#         # Look for specific quantitative columns like sales and quantity
#         for quant in quantitative_columns:
#             for cat in categorical_columns:
#                 if quant in mapped_columns and cat in mapped_columns:
#                     # Build the SQL query with optional brand filtering
#                     if brand_name:
#                         sql_query = f"SELECT {cat}, AVG(sales) as average_sales, AVG(quantity) as average_quantity FROM {table_name} WHERE brand = '{brand_name}' GROUP BY {cat}, season"
#                         nat_lang_query = f"Average sales and quantity by {cat} and season where brand is '{brand_name}'"
#                     else:
#                         sql_query = f"SELECT {cat}, AVG(sales) as average_sales, AVG(quantity) as average_quantity FROM {table_name} GROUP BY {cat}, season"
#                         nat_lang_query = f"Average sales and quantity by {cat} and season"

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
# Extract any filters from the user input, such as brand or category conditions

def extract_filters(tokens, column_names):
    filters = {}
    
    # Check for filters related to any column in the user input
    for token in tokens:
        for col in column_names:
            if col.lower() in token.lower():
                # Assume that the token represents a filter condition, e.g., "category = 'Furniture'"
                if "=" in token:
                    column, value = token.split("=")
                    filters[col] = value.strip("'").strip('"')
                elif ">" in token or "<" in token:
                    # Capture numeric range filters like price > 100
                    filters[col] = token
    return filters

def generate_sql_query(user_input, column_names, table_name, dataframe):
    # Clean and tokenize the user input
    tokens = process_input(user_input)

    # Map the tokens to actual column names
    mapped_columns = [col for col in column_names if col.lower() in tokens]

    if not mapped_columns:
        return "No matching columns found in your input. Please try again.", None

    # Categorize columns based on the dataframe
    categorical_columns, quantitative_columns = categorize_columns(dataframe)

    # Extract any filters from the user input, such as specific column values (e.g., brand, category)
    filters = extract_filters(tokens, column_names)

    # Check for common aggregation keywords like "average", "avg"
    if 'average' in tokens or 'avg' in tokens:
        aggregation_type = 'AVG'
    else:
        return "Aggregation type not recognized", None

    # Find relevant columns from user input
    group_by_columns = [col for col in column_names if col.lower() in tokens]
    
    # If no group-by columns found, assume the user meant to group by a category (like 'category')
    if not group_by_columns:
        group_by_columns = ['category']

    # Build the SQL SELECT part of the query
    select_columns = [f"{aggregation_type}({col}) AS average_{col}" for col in quantitative_columns]

    # Build the SQL WHERE part for any filters
    where_clauses = []
    if "where" in tokens:
        # Extract filter conditions (like 'price > 100')
        where_start = tokens.index('where') + 1
        where_conditions = " ".join(tokens[where_start:])
        where_clauses.append(where_conditions)

    # Construct the final SQL query
    sql_query = f"SELECT {', '.join(group_by_columns)}, {', '.join(select_columns)} FROM {table_name}"

    if where_clauses:
        sql_query += " WHERE " + " AND ".join(where_clauses)

    sql_query += f" GROUP BY {', '.join(group_by_columns)}"
    
    # Return a basic natural language version of the query and the SQL query
    nat_lang_query = f"Average {', '.join(quantitative_columns)} by {', '.join(group_by_columns)}"
    if where_clauses:
        nat_lang_query += " where " + " and ".join(where_clauses)

    return nat_lang_query, sql_query
    
    # Handle filters like "greater than", "less than", etc.
    if "greater" in tokens or "less" in tokens or "equals" in tokens:
        for quant in quantitative_columns:
            if quant in mapped_columns:
                # Extract condition
                operator = None
                if "greater" in tokens:
                    operator = ">"
                elif "less" in tokens:
                    operator = "<"
                elif "equals" in tokens:
                    operator = "="

                if operator:
                    condition_start = tokens.index(operator) + 1
                    value = tokens[condition_start]
                    sql_query = f"""
                        SELECT *
                        FROM {table_name}
                        WHERE {quant} {operator} {value}
                    """
                    nat_lang_query = f"Rows where {quant} {operator} {value}"
                    return nat_lang_query, sql_query

    # Handle "top N <A> by <B>"
    if "top" in tokens and "by" in tokens:
        for quant in quantitative_columns:
            for cat in categorical_columns:
                if quant in tokens and cat in tokens:
                    n_value = 5  # Default top 5, or extract from user input
                    if "top" in tokens:
                        idx = tokens.index("top")
                        if idx + 1 < len(tokens) and tokens[idx + 1].isdigit():
                            n_value = int(tokens[idx + 1])

                    sql_query = f"""
                        SELECT {cat}, SUM({quant}) as total_{quant}
                        FROM {table_name}
                        GROUP BY {cat}
                        ORDER BY total_{quant} DESC
                        LIMIT {n_value}
                    """
                    nat_lang_query = f"Top {n_value} {cat} by {quant}"
                    return nat_lang_query, sql_query

    # Example query pattern: "total <A> by <B>"
    if "total" in tokens or "sum" in tokens:
        for quant in quantitative_columns:
            for cat in categorical_columns:
                if quant in tokens and cat in tokens:
                    sql_query = f"SELECT {cat}, SUM({quant}) as total_{quant} FROM {table_name} GROUP BY {cat}"
                    nat_lang_query = f"Total {quant} by {cat}"
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
                result = collection.aggregate([{"$group": {"_id": "$" + nat_lang_query, "total": {"$sum": "$" + nat_lang_query}}}])
                st.write("**Query Result from MongoDB:**")
                st.write(result)
        else:
            st.write(nat_lang_query)
