import streamlit as st
import pandas as pd
import sqlite3
from pymongo import MongoClient
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
from streamlit_extras.stylable_container import stylable_container
import textwrap as tw

from sql_sample_queries import categorize_columns, generate_sample_queries, generate_construct_queries
from mongo_queries import get_quant_range, infer_types, get_sample_mongo_specific, get_sample_mongo_gen, get_mongo_queries_nat

# Download NLTK resources
def download_nltk_resources():
    try:
        nltk.download('punkt')  # Required for tokenization
        nltk.download('stopwords')  # For stopword removal
        nltk.download('wordnet')  # For lemmatization
        st.write("NLTK resources downloaded successfully.")
    except Exception as e:
        st.write(f"Error downloading NLTK resources: {e}")

st.markdown(
    """
    <style>
    .streamlit-code {
        white-space: pre-wrap;  /* Allow line wrapping */
        word-break: break-word; /* Break long words */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Call the download function
download_nltk_resources()

# Define the basic tokenizer (splitting on spaces and lowering text)
def basic_tokenizer(text):
    return text.lower().split()

# Initialize MongoDB
mongo_client = MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["chatdb"]
sqldb_list = []
mongodb_list = []

# Helper function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'json'}

# Function to store data in MongoDB
def store_in_mongodb(data, json_name):
    collectionName = json_name[:-5]
    collection = mongo_db[collectionName]
    collection.drop()  # Clear old data before inserting new
    collection.insert_many(data.to_dict(orient='records'))
    mongodb_list.append(collectionName)
    return collectionName

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
            numeric, categorical, nested, unique, data_ = infer_types(data)
            uploaded_columns = make_sql_db(data, filename)
            table_name = filename[:-4]
        else:
            data = pd.read_json(file)
            numeric, categorical, nested, unique, data = infer_types(data)
            range_vals = get_quant_range(data, numeric)
            uploaded_columns = numeric
            collection_name = store_in_mongodb(data, filename)

        st.markdown(f"**Uploaded Successfully!**  \n\nQuantitative columns in your data: {numeric}  \n\nCategorical columns in your data: {categorical} \n\nUnique columns in your data: {unique}")
    else:
        st.write("**Unsupported file type. Please upload a CSV or JSON file.**")

# Chat interface
st.write("---")
st.write("**Chat with ChatDB:**")
user_input = st.text_input("Type your query here:")

if user_input and uploaded_columns:
    # sql case
    if filename.endswith('.csv'):
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
            nat_lang_query, sql_query = generate_sql_query(user_input, uploaded_columns, table_name)

            if sql_query:
                st.write(f"**Natural Language Query:** {nat_lang_query}")
                st.code(sql_query)
                conn = sqlite3.connect("chatdb_sql.db")
                result = pd.read_sql_query(sql_query, conn)
                st.write("**Query Result from SQLite:**")
                st.dataframe(result)
            else:
                st.write(nat_lang_query)
    # mongo db case
    if filename.endswith('.json'):
        # Execute the query on MongoDB database
        collection = mongo_db[collection_name]
        tokens = process_input(user_input)
        example_with_spec_terms = ['aggregate', 'group', 'find', 'where', 'retrieve']
        example_with_spec = ["with", "using"]
        example_general = ["example", "sample"]
        example_nat_lang = ["sum", "total", "average", "mean", "greater", "more", "than", "above", "less", "fewer", "below", "count", "number", "counts"]
        result = []
        # general example
        if set(tokens).isdisjoint(example_with_spec+example_nat_lang):
            result = get_sample_mongo_gen(categorical, numeric, unique, range_vals, collection_name)
        # specific general
        elif set(example_with_spec_terms) & set(tokens):
            result = get_sample_mongo_specific(tokens, categorical, numeric, unique, range_vals, collection_name)
        # nat language example
        elif set(tokens) & set(example_nat_lang):
            result = get_mongo_queries_nat(tokens, categorical, numeric, unique, range_vals, collection_name)
        if result:
            if isinstance(result, dict):
                for key, value in result.items():
                    data = list(value[0])
                    natural_lang = value[2]
                    query_code = value[1]
                    type_ = value[3]
                    type_ = type_.capitalize()
                    st.write(f"**{type_}:**\n\n {natural_lang}")
                    # https://discuss.streamlit.io/t/st-code-on-multiple-lines/50511/9
                    with stylable_container(
                        "codeblock",
                        """
                        code {
                            white-space: pre-wrap !important;
                        }
                        """,
                    ):
                        st.code(
                            query_code
                        )
                    result_df = pd.DataFrame(data)
                    st.write("**Query Result from MongoDB:**")
                    st.dataframe(result_df)
               # do dict stuff
            else:
                data = list(result[0])
                query_code = result[1]
                nat_query = result[2]
                type_ = result[3]
                st.write(f"**{type_}:**\n\n {nat_query}")
                result_df = pd.DataFrame(result)
                with stylable_container(
                        "codeblock",
                        """
                        code {
                            white-space: pre-wrap !important;
                        }
                        """,
                    ):
                        st.code(
                            query_code
                        )
                result_df = pd.DataFrame(data)
                st.write("**Query Result from MongoDB:**")
                st.dataframe(result_df)
        else:
            st.write("Sorry, none of our queries matched your request. Please try again")
elif user_input:
    st.write("Please upload a dataset first.")

# Display chat history
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
