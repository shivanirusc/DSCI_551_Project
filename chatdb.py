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

from sql_queries import categorize_columns, generate_sample_queries, generate_construct_queries, execute_query

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

def generate_combined_tokens(tokens):
    combined_tokens = [
        ' '.join(tokens[i:j+1]) for i in range(len(tokens)) for j in range(i, len(tokens))
    ]
    combined_tokens = [token.replace(' ', '_').lower() for token in combined_tokens]
    return combined_tokens

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

    # Step 2: Dynamically categorize columns in the DataFrame
    categorical_columns, quantitative_columns = categorize_columns(data)


    # Step 3: Combine tokens to handle multi-word column names
    combined_tokens = [' '.join(tokens[i:j+1]) for i in range(len(tokens)) for j in range(i, len(tokens))]
    combined_tokens = [token.replace(' ', '_').lower() for token in combined_tokens]  # Format like column names

    # Step 4: Handle Top Aggregation Queries dynamically
    if any(word in tokens for word in ["highest", "top"]):
        quant_col = None
        cat_col = None

        # Match combined tokens to quantitative columns (e.g., "profit_margin")
        for quant in quantitative_columns:
            if quant.lower() in combined_tokens:
                quant_col = quant
                break

        # Match combined tokens to categorical columns (e.g., "region")
        for cat in categorical_columns:
            if cat.lower() in combined_tokens:
                cat_col = cat
                break

        # Generate SQL query if both column mappings are found
        if quant_col and cat_col:
            sql_query = f"SELECT {cat_col}, MAX({quant_col}) as max_{quant_col} FROM {table_name} GROUP BY {cat_col}"
            nat_lang_query = f"Top {quant_col} by {cat_col}"
            return nat_lang_query, sql_query
    
    # Handle sum and total queries
    if "sum" in tokens or "total" in tokens:
        column = map_columns(combined_tokens, quantitative_columns)  # Identify the quantitative column
        group_by_column = map_columns(combined_tokens, categorical_columns)  # Identify the categorical column
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

    # Handle custom aggregations
    if "total" in tokens and "average" in tokens:
        combined_tokens = generate_combined_tokens(tokens)
        sum_column = next((col for col in quantitative_columns if col.lower() in combined_tokens), None)
        avg_column = next((col for col in quantitative_columns if col.lower() in combined_tokens), None)
        group_by_column = next((col for col in categorical_columns if col.lower() in combined_tokens), None)

        if sum_column and avg_column and group_by_column:
            sql_query = (f"SELECT {group_by_column}, SUM({sum_column}) as total_{sum_column}, "
                         f"AVG({avg_column}) as avg_{avg_column} FROM {table_name} GROUP BY {group_by_column}")
            nat_lang_query = f"Total {sum_column} and average {avg_column} grouped by {group_by_column}"
            return nat_lang_query, sql_query


    # Fallback in case no match is found
    return "Query could not be interpreted. Please try rephrasing.", ""

# Streamlit app setup
st.title("ChatDB: Interactive Query Assistant 🤖")
st.write("Upload your dataset (CSV or JSON), and ask ChatDB to generate queries for your data using natural language.")

# Sidebar for instructions
st.sidebar.title("Instructions 📖")
st.sidebar.markdown("""
1. Upload your dataset (CSV or JSON).
2. Enter a natural language query in the input box.
3. View the generated query, explanation, and results.
4. Chat history is displayed for reference.
""")

# File upload section
st.sidebar.subheader("Upload Dataset")
file = st.sidebar.file_uploader("Choose a CSV or JSON file:", type=["csv", "json"])
uploaded_columns = []
table_name = ""
data = None

if file:
    filename = file.name
    filetype = "csv" if filename.endswith(".csv") else "json"
    if allowed_file(filename):
        if filetype == "csv":
            data = pd.read_csv(file)
            uploaded_columns = make_sql_db(data, filename)
            table_name = filename[:-4]
        elif filetype == "json":
            data = pd.read_json(file)
            uploaded_columns = store_in_mongodb(data, filename)
            table_name = filename[:-5]
        st.success(f"Dataset uploaded successfully! Columns in your data: {uploaded_columns}")
    else:


        st.error("Unsupported file type. Please upload a CSV or JSON file.")

# Chat interface only if dataset is loaded
if data is not None:
    st.write("---")
    st.subheader("Chat with ChatDB 💬")
    user_input = st.text_input("Type your query here:")

    if user_input and uploaded_columns:
        nat_lang_query, query = generate_sql_query(user_input, uploaded_columns, table_name, data)

        # Add to chat history only if a valid query is generated
        if query:
            # Initialize chat history if not already present
            if "chat_history" not in st.session_state:
                st.session_state["chat_history"] = []

            # Add current query and response to chat history
            st.session_state["chat_history"].append({
                "user_input": user_input,
                "nat_lang_query": nat_lang_query,
                "query": query
            })
        
        st.write("---")
        st.subheader("Generated Query 🔍")
        if query:
            st.markdown(f"**Natural Language Interpretation:** `{nat_lang_query}`")
            st.code(query)
        else:
            st.error("No query generated. Please refine your input.")
        
        # Explanation Section
        st.write("---")
        st.subheader("Query Explanation 📝")
        if query:
            if filetype == "csv" and isinstance(query, str):
                explanation = "This SQL query performs the following operations:\n\n"
                if "GROUP BY" in query:
                    explanation += "- Groups the data by one or more categorical columns.\n"
                if "SUM" in query:
                    explanation += "- Calculates the total for a specified numerical column.\n"
                if "MAX" in query:
                    explanation += "- Finds the maximum value in a numerical column.\n"
                if "AVG" in query:
                    explanation += "- Computes the average value of a numerical column.\n"
                if "WHERE" in query:
                    explanation += "- Filters the data based on specific conditions (e.g., 'less than' or 'greater than').\n"
                explanation += f"\nThe query retrieves data from the `{table_name}` table."
            elif filetype == "json" and isinstance(query, list):
                explanation = "This NoSQL query uses an aggregation pipeline to:\n\n"
                explanation += "- Filter documents based on the specified criteria.\n"
                explanation += "- Group the data and compute metrics such as sum, average, or maximum.\n"
                explanation += f"\nThe query operates on the `{table_name}` collection in MongoDB."
            else:
                explanation = "Unable to explain the query."
            st.write(explanation)
        else:
            st.error(nat_lang_query)

        # Results Section
        st.write("---")
        st.subheader("Query Results 📊")
        if query:
            try:
                if filetype == "csv":
                    conn = sqlite3.connect("chatdb_sql.db")
                    result = pd.read_sql_query(query, conn)
                    st.dataframe(result)
                elif filetype == "json":
                    collection = mongo_db[table_name]
                    result = list(collection.aggregate(query))
                    result_df = pd.DataFrame(result)
                    st.dataframe(result_df)
            except Exception as e:
                st.error(f"Error executing query: {e}")
        else:
            st.error("No results generated for this query.")

    # Chat History Section
    st.write("---")
    st.subheader("Chat History 🕒")
    if "chat_history" in st.session_state:
        for idx, chat in enumerate(st.session_state["chat_history"]):
            user_query = chat.get("user_input", "Unknown query")
            response = chat.get("nat_lang_query", "Unknown response")
            generated_query = chat.get("query", "No query generated")
            st.markdown(f"**Query {idx + 1}:**")
            st.write(f"**You:** {user_query}")
            st.write(f"**ChatDB:** {response}")
            st.code(generated_query)
else:
    st.write("Please upload a dataset to start interacting with ChatDB.")
