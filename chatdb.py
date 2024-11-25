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

def tokenize_with_operators(user_input):
    # Regular expression to capture words, numbers, comparison operators, and logical operators
    pattern = r'\b(and|or)\b|[a-zA-Z_]+|[<>!=]+|\d+'

    # Apply regex to tokenize the input
    tokens = re.findall(pattern, user_input.lower())

    return [token.strip() for token in tokens if token.strip()]

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
        
        if quant_col and cat_col:
            sql_query = f"SELECT {cat_col}, AVG({quant_col}) as average_{quant_col} FROM {table_name} GROUP BY {cat_col}"
            nat_lang_query = f"Average {quant_col} by {cat_col}"
            return nat_lang_query, sql_query

    # Handle 'maximum' or 'max' queries
    if any(word in tokens for word in ["maximum", "max"]):
        # Match quantitative column
        matched_column = None
        for quant in quantitative_columns:
            if any(token in quant.lower() for token in tokens):
                matched_column = quant
                break  # Exit loop once a match is found
            
        if matched_column:
            sql_query = f"SELECT '{matched_column}', MAX({matched_column}) as max_{matched_column} FROM {table_name}"
            nat_lang_query = f"Maximum {matched_column}"
            return nat_lang_query, sql_query

    # Handle 'minimum' or 'min' queries
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

    # Handle 'greater', 'less', 'equal', 'not equal', 'between'
    tokens1 = tokenize_with_operators(user_input)
    st.write(f"Tokens1 extracted: {tokens1}")
    # Initialize components for SQL conditions
    conditions = []
    operators = {
        "less": "<",
        "greater": ">",
        "equal": "=",
        "not equal": "!=",
        "between": "BETWEEN"
    }
    conjunctions = ["and", "or"]

    i = 0
    while i < len(tokens):
        token = tokens[i]
        
        # Match a quantitative column
        matched_column = None
        for quant in quantitative_columns:
            if token in quant.lower():
                matched_column = quant
                break

        if matched_column:
            operator = None
            if i + 1 < len(tokens) and tokens[i + 1] in operators:
                operator = operators[tokens[i + 1]]
                i += 1  # Move to operator token
            
            if operator and operator != "BETWEEN":
                # Handle single value operators (<, >, =, !=)
                if i + 1 < len(tokens) and tokens[i + 1].isdigit():
                    value = tokens[i + 1]
                    conditions.append(f"{matched_column} {operator} {value}")
                    i += 1  # Move past the value
            elif operator == "BETWEEN":
                # Handle BETWEEN operator with two values
                if i + 2 < len(tokens) and tokens[i + 1].isdigit() and tokens[i + 2].isdigit():
                    value1 = tokens[i + 1]
                    value2 = tokens[i + 2]
                    conditions.append(f"{matched_column} {operator} {value1} AND {value2}")
                    i += 2  # Move past the two values
        
        # Handle conjunctions
        elif token in conjunctions:
            if conditions and token.lower() in conjunctions:
                conditions.append(token1.upper())
        
        i += 1
        
    if "or" in tokens1:
        where_clause = " OR ".join(conditions)  # Join with OR if explicit OR exists
    elif "and" in tokens1:
        where_clause = " AND ".join(conditions)  # Default to AND
    else:
        where_clause = " ".join(conditions)

    if where_clause:
        sql_query = f"SELECT * FROM {table_name} WHERE {where_clause}"
        nat_lang_query = f"Rows where {where_clause}"
        print(f"Generated query: {sql_query}")
        return nat_lang_query, sql_query

    # Support for nested conditions
    nested_conditions = []
    temp_condition = []
    
    # Handle nested conditions and logical operators
    if "(" in tokens or ")" in tokens:
        i = 0
        while i < len(tokens):
                token = tokens[i]
        
                if token == "(":
                    # Start a nested condition
                    if temp_condition:
                        nested_conditions.append(" ".join(temp_condition))
                        temp_condition = []
                    nested_conditions.append("(")
                elif token == ")":
                    # Close a nested condition
                    if temp_condition:
                        nested_conditions.append(" ".join(temp_condition))
                        temp_condition = []
                    nested_conditions.append(")")
                else:
                    temp_condition.append(token)
        
                i += 1

    # Join the nested conditions into a WHERE clause
    where_clause = " ".join(nested_conditions)
    sql_query = f"SELECT * FROM {table_name} WHERE {where_clause}"
    nat_lang_query = f"Rows where {where_clause}"
    return nat_lang_query, sql_query

    # Handle wildcard searches
    if "contains" in tokens or "like" in tokens:
        for cat in categorical_columns:
            if any(token in cat.lower() for token in tokens):
                value = [token for token in tokens if token not in ["contains", "like"]]
                search_value = value[-1] if value else ""
                sql_query = f"SELECT * FROM {table_name} WHERE {cat} LIKE '%{search_value}%'"
                nat_lang_query = f"Rows where {cat} contains {search_value}"
                return nat_lang_query, sql_query

    # Handle custom aggregations
    if "total" in tokens and "average" in tokens:
        sum_column = map_columns(tokens, quantitative_columns)
        avg_column = map_columns(tokens, quantitative_columns)
        group_by_column = map_columns(tokens, categorical_columns)

    if sum_column and avg_column and group_by_column:
        sql_query = f"SELECT {group_by_column}, SUM({sum_column}) as total_{sum_column}, AVG({avg_column}) as avg_{avg_column} FROM {table_name} GROUP BY {group_by_column}"
        nat_lang_query = f"Total {sum_column} and average {avg_column} grouped by {group_by_column}"
        return nat_lang_query, sql_query

    # Handle range queries
    ranges = []
    i = 0
    while i < len(tokens):
        if tokens[i] == "between" and i + 2 < len(tokens):
            column = map_columns([tokens[i - 1]], quantitative_columns)
            if column and tokens[i + 1].isdigit() and tokens[i + 2].isdigit():
                start = tokens[i + 1]
                end = tokens[i + 2]
                ranges.append(f"{column} BETWEEN {start} AND {end}")
                i += 2
        i += 1

    if ranges:
        where_clause = " AND ".join(ranges)
        sql_query = f"SELECT * FROM {table_name} WHERE {where_clause}"
        nat_lang_query = f"Rows where {where_clause}"
        return nat_lang_query, sql_query

    # Handle Top-N Queries
     if "top" in tokens:
        for quant in quantitative_columns:
            # Match singular/plural forms of the quantitative column
            if any(token in quant.lower() for token in tokens):
                # Extract N from tokens (default to 5 if not specified)
                top_n = next((int(token) for token in tokens if token.isdigit()), 5)
                sql_query = f"SELECT * FROM {table_name} ORDER BY {quant} DESC LIMIT {top_n}"
                nat_lang_query = f"Top {top_n} products by {quant}"
                print(f"Generated SQL Query: {sql_query}")
                return nat_lang_query, sql_query

    # Handle data filtering
    if "from" in tokens and "to" in tokens:
        date_column = "sale_date"  # Example fixed date column
        date_indices = [i for i, token in enumerate(tokens) if token in ["from", "to"]]
        if len(date_indices) == 2 and date_indices[1] > date_indices[0]:
            start_date = tokens[date_indices[0] + 1]
            end_date = tokens[date_indices[1] + 1]
            sql_query = f"SELECT * FROM {table_name} WHERE {date_column} BETWEEN '{start_date}' AND '{end_date}'"
            nat_lang_query = f"Rows where {date_column} is between {start_date} and {end_date}"
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
