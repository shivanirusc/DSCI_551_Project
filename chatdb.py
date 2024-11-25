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

def extract_join_info(user_input, uploaded_columns):
    # Define possible join types
    join_types = ["INNER", "LEFT", "RIGHT"]
    
    # Initialize join information
    join_type = None
    join_table = None
    join_columns = None
    
    # Detect join type (e.g., INNER, LEFT, RIGHT)
    for jt in join_types:
        if jt in user_input.upper():
            join_type = jt
            break  # stop once we find the join type
    
    # Extract the tables involved in the join (e.g., "product_data", "category_data")
    table_match = re.search(r'(\w+)\s+with\s+(\w+)', user_input)
    if table_match:
        # The first table in the main SELECT statement is assumed to be the base table
        table_name = table_match.group(1)
        join_table = table_match.group(2)
    else:
        # Default case if no join tables are explicitly mentioned
        table_name = "product_data"  # Assuming default base table if not provided
    
    # Extract the column names for the join condition (e.g., "product_data.category = category_data.category")
    join_condition_match = re.search(r"on\s+(\w+\.\w+)\s*=\s*(\w+\.\w+)", user_input)
    if join_condition_match:
        join_columns = (join_condition_match.group(1), join_condition_match.group(2))
    
    return join_type, table_name, join_table, join_columns

# Function to generate SQL queries based on user input
def generate_sql_query(user_input, column_names, table_name, dataframe):
    # Clean and tokenize the user input
    tokens = process_input(user_input)
    
    # Map the tokens to actual column names
    mapped_columns = [col for col in column_names if col.lower() in tokens]

    if not mapped_columns:
        return "No matching columns found in your input. Please try again.", None

    # Categorize columns based on the dataframe
    categorical_columns, quantitative_columns = categorize_columns(dataframe)

    # # Example query pattern: "total <A> by <B>"
    # if "total" in tokens or "sum" in tokens:
    #     for quant in quantitative_columns:
    #         for cat in categorical_columns:
    #             if quant in tokens and cat in tokens:
    #                 sql_query = f"SELECT {cat}, SUM({quant}) as total_{quant} FROM {table_name} GROUP BY {cat}"
    #                 nat_lang_query = f"Total {quant} by {cat}"
    #                 return nat_lang_query, sql_query

    # # Example pattern: "average <A> by <B>"
    # if "average" in tokens or "avg" in tokens:
    #     # Look for specific quantitative columns like sales and quantity
    #     for quant in quantitative_columns:
    #         for cat in categorical_columns:
    #             if quant in mapped_columns and cat in mapped_columns:
    #                 # Build the SQL query with optional brand filtering
    #                 if brand_name:
    #                     sql_query = f"SELECT {cat}, AVG(sales) as average_sales, AVG(quantity) as average_quantity FROM {table_name} WHERE brand = '{brand_name}' GROUP BY {cat}, season"
    #                     nat_lang_query = f"Average sales and quantity by {cat} and season where brand is '{brand_name}'"
    #                 else:
    #                     sql_query = f"SELECT {cat}, AVG(sales) as average_sales, AVG(quantity) as average_quantity FROM {table_name} GROUP BY {cat}, season"
    #                     nat_lang_query = f"Average sales and quantity by {cat} and season"

    #                 return nat_lang_query, sql_query

    # # Example pattern: "maximum <A> by <B>"
    # if "maximum" in tokens or "max" in tokens:
    #     for quant in quantitative_columns:
    #         for cat in categorical_columns:
    #             if quant in tokens and cat in tokens:
    #                 sql_query = f"SELECT {cat}, MAX({quant}) as max_{quant} FROM {table_name} GROUP BY {cat}"
    #                 nat_lang_query = f"Maximum {quant} by {cat}"
    #                 return nat_lang_query, sql_query

    # # Example pattern: "count of <A> by <B>"
    # if "count" in tokens or "total" in tokens:
    #     for cat in categorical_columns:
    #         if cat in tokens:
    #             sql_query = f"SELECT {cat}, COUNT(*) as count_{cat} FROM {table_name} GROUP BY {cat}"
    #             nat_lang_query = f"Count of {cat}"
    #             return nat_lang_query, sql_query

    # # Example pattern: "total <A> where <B>"
    # if "where" in tokens:
    #     for quant in quantitative_columns:
    #         if quant in tokens:
    #             condition = ' '.join(tokens[tokens.index("where")+1:])
    #             sql_query = f"SELECT SUM({quant}) as total_{quant} FROM {table_name} WHERE {condition}"
    #             nat_lang_query = f"Total {quant} where {condition}"
    #             return nat_lang_query, sql_query

    # # Example pattern: "top N <A> by <B>"
    # if "top" in tokens and "by" in tokens:
    #     for quant in quantitative_columns:
    #         for cat in categorical_columns:
    #             if quant in tokens and cat in tokens:
    #                 n_value = 5  # Default top 5, could be extracted from input if specified
    #                 sql_query = f"SELECT {cat}, SUM({quant}) as total_{quant} FROM {table_name} GROUP BY {cat} ORDER BY total_{quant} DESC LIMIT {n_value}"
    #                 nat_lang_query = f"Top {n_value} {cat} by {quant}"
    #                 return nat_lang_query, sql_query
    # Example query pattern: "total <A> by <B>"
   # Handle aggregate functions: SUM, AVG, MAX, COUNT
    aggregate_keywords = {
        "sum": "SUM", "total": "SUM", "average": "AVG", "avg": "AVG",
        "max": "MAX", "count": "COUNT"
    }

    for keyword, agg_func in aggregate_keywords.items():
        if keyword in tokens:
            for quant in quantitative_columns:
                for cat in categorical_columns:
                    if quant in tokens and cat in tokens:
                        sql_query = f"SELECT {cat}, {agg_func}({quant}) as {agg_func.lower()}_{quant} FROM {table_name} GROUP BY {cat}"
                        nat_lang_query = f"{agg_func} {quant} by {cat}"
                        return nat_lang_query, sql_query

    # Handle "WHERE" conditions with multiple tokens
    if "where" in tokens:
        condition = ' '.join(tokens[tokens.index("where")+1:])
        sql_query = f"SELECT * FROM {table_name} WHERE {condition}"
        nat_lang_query = f"Data where {condition}"
        return nat_lang_query, sql_query

    # Handle "TOP N" queries
    if "top" in tokens and "by" in tokens:
        for quant in quantitative_columns:
            for cat in categorical_columns:
                if quant in tokens and cat in tokens:
                    n_value = 5  # Default to top 5; you can extract this dynamically from the input
                    sql_query = f"SELECT {cat}, SUM({quant}) as total_{quant} FROM {table_name} GROUP BY {cat} ORDER BY total_{quant} DESC LIMIT {n_value}"
                    nat_lang_query = f"Top {n_value} {cat} by {quant}"
                    return nat_lang_query, sql_query
    
    # 1. Total (SUM) queries
    # if "total" in tokens or "sum" in tokens:
    #     for quant in quantitative_columns:
    #         for cat in categorical_columns:
    #             if quant in tokens and cat in tokens:
    #                 sql_query = f"SELECT {cat}, SUM({quant}) as total_{quant} FROM {table_name} GROUP BY {cat}"
    #                 nat_lang_query = f"Total {quant} by {cat}"
    #                 return nat_lang_query, sql_query

    # 2. Average (AVG) queries
    # if "average" in tokens or "avg" in tokens:
    #     for quant in quantitative_columns:
    #         for cat in categorical_columns:
    #             if quant in tokens and cat in tokens:
    #                 sql_query = f"SELECT {cat}, AVG({quant}) as average_{quant} FROM {table_name} GROUP BY {cat}"
    #                 nat_lang_query = f"Average {quant} by {cat}"
    #                 return nat_lang_query, sql_query

    # # 3. Maximum (MAX) queries
    # if "maximum" in tokens or "max" in tokens:
    #     for quant in quantitative_columns:
    #         for cat in categorical_columns:
    #             if quant in tokens and cat in tokens:
    #                 sql_query = f"SELECT {cat}, MAX({quant}) as max_{quant} FROM {table_name} GROUP BY {cat}"
    #                 nat_lang_query = f"Maximum {quant} by {cat}"
    #                 return nat_lang_query, sql_query

    # 4. Count queries (COUNT)
    # if "count" in tokens:
    #     for cat in categorical_columns:
    #         if cat in tokens:
    #             sql_query = f"SELECT {cat}, COUNT(*) as count_{cat} FROM {table_name} GROUP BY {cat}"
    #             nat_lang_query = f"Count of {cat}"
    #             return nat_lang_query, sql_query

    # 5. Conditional Total queries (SUM WHERE)
    # if "where" in tokens:
    #     for quant in quantitative_columns:
    #         if quant in tokens:
    #             condition = ' '.join(tokens[tokens.index("where")+1:])
    #             sql_query = f"SELECT SUM({quant}) as total_{quant} FROM {table_name} WHERE {condition}"
    #             nat_lang_query = f"Total {quant} where {condition}"
    #             return nat_lang_query, sql_query

    # # 6. Top N queries (e.g., "Top N sales by category")
    # if "top" in tokens and "by" in tokens:
    #     for quant in quantitative_columns:
    #         for cat in categorical_columns:
    #             if quant in tokens and cat in tokens:
    #                 n_value = 5  # Default top 5, can be modified to extract from user input
    #                 sql_query = f"SELECT {cat}, SUM({quant}) as total_{quant} FROM {table_name} GROUP BY {cat} ORDER BY total_{quant} DESC LIMIT {n_value}"
    #                 nat_lang_query = f"Top {n_value} {cat} by {quant}"
    #                 return nat_lang_query, sql_query 

    # 7. Complex queries with multiple conditions (WHERE + AND)
    if "where" in tokens and "and" in tokens:
        conditions = ' AND '.join(tokens[tokens.index("where")+1:])
        sql_query = f"SELECT * FROM {table_name} WHERE {conditions}"
        nat_lang_query = f"Data where {conditions}"
        return nat_lang_query, sql_query

    # 8. Grouped and filtered queries (e.g., Group by category and filter by price)
    if "group" in tokens and "by" in tokens and "where" in tokens:
        group_column = [col for col in categorical_columns if col.lower() in tokens]
        condition_column = [col for col in quantitative_columns if col.lower() in tokens]
        if group_column and condition_column:
            condition = ' '.join(tokens[tokens.index("where")+1:])
            sql_query = f"SELECT {', '.join(group_column)}, SUM({condition_column[0]}) as total_{condition_column[0]} FROM {table_name} WHERE {condition} GROUP BY {', '.join(group_column)}"
            nat_lang_query = f"Total {condition_column[0]} by {', '.join(group_column)} where {condition}"
            return nat_lang_query, sql_query
    
    # Extract the join type, base table, join table, and join columns from the user input
    join_type, table_name, join_table, join_columns = extract_join_info(user_input, uploaded_columns)
    
    # Construct the SQL query based on the extracted information
    if join_type and join_table and join_columns:
        sql_query = f"SELECT * FROM {table_name} {join_type} JOIN {join_table} ON {join_columns[0]} = {join_columns[1]}"
        return user_input, sql_query
    else:
        sql_query = f"SELECT * FROM {table_name}"
        return user_input, sql_query
    
    
    # Initialize join_type, join_table, and join_columns
    join_type, join_table, join_columns = extract_join_info(user_input, uploaded_columns)
    
    # Construct the SQL query based on extracted information
    if join_type and join_table and join_columns:
        sql_query = f"SELECT * FROM {table_name} {join_type} JOIN {join_table} ON {join_columns[0]} = {join_columns[1]}"
        return user_input, sql_query
    else:
        sql_query = f"SELECT * FROM {table_name}"
        return user_input, sql_query

    # 2. Aggregation with JOIN (e.g., SUM, AVG)
    if "total" in tokens or "sum" in tokens or "average" in tokens or "avg" in tokens:
        for quant in quantitative_columns:
            for cat in categorical_columns:
                if quant in tokens and cat in tokens:
                    if join_table and join_columns:
                        join_condition = f"ON {table_name}.{join_columns[0]} = {join_table}.{join_columns[1]}"
                        sql_query = f"SELECT {cat}, SUM({quant}) as total_{quant} FROM {table_name} {join_type} {join_table} {join_condition} GROUP BY {cat}"
                        nat_lang_query = f"Total {quant} by {cat} joined with {join_table}"
                    else:
                        sql_query = f"SELECT {cat}, SUM({quant}) as total_{quant} FROM {table_name} GROUP BY {cat}"
                        nat_lang_query = f"Total {quant} by {cat}"
                    return nat_lang_query, sql_query

    # 3. Filtering with JOIN (e.g., WHERE clause)
    if "where" in tokens:
        for quant in quantitative_columns:
            if quant in tokens:
                condition = ' '.join(tokens[tokens.index("where")+1:])
                if join_table and join_columns:
                    join_condition = f"ON {table_name}.{join_columns[0]} = {join_table}.{join_columns[1]}"
                    sql_query = f"SELECT {quant}, SUM({quant}) as total_{quant} FROM {table_name} {join_type} {join_table} {join_condition} WHERE {condition}"
                    nat_lang_query = f"Total {quant} where {condition} joined with {join_table}"
                else:
                    sql_query = f"SELECT {quant}, SUM({quant}) as total_{quant} FROM {table_name} WHERE {condition}"
                    nat_lang_query = f"Total {quant} where {condition}"
                return nat_lang_query, sql_query

    # 4. JOIN with multiple conditions (AND)
    if "and" in tokens and "join" in tokens:
        if join_table and join_columns:
            join_condition = f"ON {table_name}.{join_columns[0]} = {join_table}.{join_columns[1]}"
            conditions = ' AND '.join(tokens[tokens.index("where")+1:])
            sql_query = f"SELECT * FROM {table_name} {join_type} {join_table} {join_condition} WHERE {conditions}"
            nat_lang_query = f"Join {table_name} with {join_table} on {join_columns[0]} = {join_columns[1]} where {conditions}"
            return nat_lang_query, sql_query

    # 5. Multiple JOINS (e.g., JOINs between multiple tables)
    if "join" in tokens and "and" in tokens:
        join_tables = [join_table]  # Assume multiple join tables
        # Create JOIN SQL for multiple tables
        for i in range(len(join_tables)-1):
            join_condition = f"ON {table_name}.{join_columns[i]} = {join_tables[i+1]}.{join_columns[i+1]}"
            sql_query += f" {join_type} {join_tables[i+1]} {join_condition}"
        nat_lang_query = f"Join multiple tables {table_name} and others"
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
