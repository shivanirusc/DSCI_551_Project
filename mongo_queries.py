import string
import pandas as pd
from pymongo import MongoClient
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import ssl
import random
from tabulate import tabulate
import json

# some learning notes as I have been going:
# don't cover cases where an attribute is a list or dict. These could be queries all on their own. This could be something we add for 
# complexity

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


mongo_client = MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["chat_db"]
mongodb_list = []

# Download NLTK resources
def download_nltk_resources():
    nltk.download('punkt')  # Required for tokenization
    nltk.download('stopwords')  # For stopword removal
    nltk.download('wordnet')  # For lemmatization

# Define the basic tokenizer (splitting on spaces and lowering text)
def basic_tokenizer(text):
    return text.lower().split()

def store_in_mongodb(data, json_name):
    collectionName = json_name[:-5]
    collection = mongo_db[collectionName]
    collection.drop()  # Clear old data before inserting new
    collection.insert_many(data.to_dict(orient='records'))
    mongodb_list.append(collectionName)
    return collectionName

def infer_types(df_):
    """Infers the types of each column and returns the different groups of columns, and ensures 
        column names don't have any punctuation or spaces for later querying

    Parameters
    ----------
    df_ : pandas dataframe
        The dataframe that will be ingested to either mongodb or sqlite

    Returns
    -------
    numeric_cols
        a list of strings used that are the header columns
    categorical_cols
        a list of strings used that are the header columns
    nested_cols
        a list of strings used that are the header columns
    unique_cols
        a list of strings used that are the header columns
    df_
        dataframe with the updated column names
    """
    # add functionality for columns that only have unique values
    numeric_cols = []
    categorical_cols = []
    nested_cols = []
    unique_cols = []

    # remove any punctuation that may hinder querying later
    df_.columns = [
        col.translate(str.maketrans("", "", string.punctuation)).replace(" ", "_")
        for col in df_.columns
    ]
    
    # for each column, determine if it is numeric, nested, or categorical, and for categorical if
    # it is a unique value
    for col in df_.columns:
        if df_[col].isna().all():
            continue
        first_non_na = df_[col].dropna().iloc[0]
        if isinstance(first_non_na, list) or isinstance(first_non_na, dict):
            nested_cols.append(col)
            continue
        numeric_col = pd.to_numeric(df_[col], errors='coerce')
        num_non_na = numeric_col.notna().sum()
        if num_non_na > 0 and (num_non_na / len(df_[col])) >= 0.5:
            numeric_cols.append(col)
            df_[col] = numeric_col 
        else:
            if df_[col].nunique() == len(df_[col].dropna()):
                # All non-NaN values are unique
                unique_cols.append(col)
            else:
                categorical_cols.append(col)
    # print("NUMERIC COLS", numeric_cols)
    # print("CATEGORICAL COLS", categorical_cols)
    # print("NESTED COLS", nested_cols)
    # print("UNIQUE COLS", unique_cols)
    return numeric_cols, categorical_cols, nested_cols, unique_cols, df_

def get_quant_range(df_, quant_cols):
    """Gets the range of each quantitative column

    Parameters
    ----------
    df_ : pandas dataframe
        The dataframe that will be ingested to either mongodb or sqlite
    quant_cols : list
        List of the quantitative variable names

    Returns
    -------
    range_dict
        Dictionary where the key is the quantitative column and the value is a tuple containing the (min, max) value for said column
    """
    range_dict = {}
    for col in quant_cols:
        min_val = df_[col].min()
        max_val = df_[col].max()
        range_dict[col] = (min_val, max_val)
    return range_dict


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

# main function of getting mongo queries from any sort of user input
def get_mongo_queries(user_input):
    print(None)

# get general sample queries
def get_sample_mongo_gen():
    print(None)

# get specific sample queries (ie group by)
def get_sample_mongo_specific():
    print(None)

# get query based on user natural language
def get_mongo_user_input():
    print(None)

def gen_total_query(cat_cols, quant_cols, collectionName):
    """Generates a query in the Total <A> for each <B> Category, where A is a random quantitative variable and B is a 
        random categorical variable, prints this query in natural language and mongo format, and prints the head(5) of the output

    Parameters
    ----------
    cat_cols : list
        The dataframe that will be ingested to either mongodb or sqlite
    quant_cols : list
        List of the quantitative variable names
    collectionName : str
        Name of the collection we are currently querying on
    """
    random_quant = random.choice(quant_cols)
    random_cat = random.choice(cat_cols)
    random_quant_var = f"Total {random_quant}"
    random_quant_var = random_quant_var.translate(str.maketrans('', '',
                                    string.punctuation))
    nat_language = f"Total {random_quant} for each {random_cat} category"
    collection = mongo_db[collectionName]
    query_result = collection.aggregate([{"$group": {"_id": f"${random_cat}", random_quant_var: {"$sum": f"${random_quant}"}}}, 
                                         {"$project": {f"{random_cat}": "$_id", random_quant_var: 1, "_id": 0}},{"$limit": 5}])
    query = [{"$group": {"_id": f"${random_cat}", random_quant_var: {"$sum": f"${random_quant}"}}}, 
                                         {"$project": {f"{random_cat}": "$_id", random_quant_var: 1, "_id": 0}},{"$limit": 5}]
    query_string = ', '.join(json.dumps(part) for part in query)
    # Extract headers correctly by getting the keys from the first row
    data = list(query_result)
    headers = list(data[0].keys())[::-1] if data else []
    # Ensure rows are aligned with reversed headers
    rows = [[row[h] for h in headers] for row in data]
    # output
    print(nat_language)
    print(query_string)
    print("Below is the result of the query:")
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    

def gen_average_query(cat_cols, quant_cols, collectionName):
    """Generates a query in the Average <A> for each <B> Category, where A is a random quantitative variable and B is a 
        random categorical variable, prints this query in natural language and mongo format, and prints the head(5) of the output

    Parameters
    ----------
    cat_cols : list
        The dataframe that will be ingested to either mongodb or sqlite
    quant_cols : list
        List of the quantitative variable names
    collectionName : str
        Name of the collection we are currently querying on
    """
    random_quant = random.choice(quant_cols)
    random_cat = random.choice(cat_cols)
    random_quant_var = f"Average {random_quant}"
    random_quant_var = random_quant_var.translate(str.maketrans('', '',
                                    string.punctuation))
    nat_language = f"Average {random_quant} for each {random_cat} category"
    collection = mongo_db[collectionName]
    query_result = collection.aggregate([{"$group": {"_id": f"${random_cat}", random_quant_var: {"$avg": f"${random_quant}"}}}, 
                                         {"$project": {f"{random_cat}": "$_id", random_quant_var: 1, "_id": 0}},{"$limit": 5}])
    query = [{"$group": {"_id": f"${random_cat}", random_quant_var: {"$avg": f"${random_quant}"}}}, 
                                         {"$project": {f"{random_cat}": "$_id", random_quant_var: 1, "_id": 0}},{"$limit": 5}]
    query_string = ', '.join(json.dumps(part) for part in query)
    data = list(query_result)
    headers = list(data[0].keys())[::-1] if data else []
    rows = [[row[h] for h in headers] for row in data]
    print(nat_language)
    print(query_string)
    print("Below is the result of the query:")
    print(tabulate(rows, headers=headers, tablefmt="grid"))

def gen_counts_query(cat_cols, collectionName):
    """Generates a query in the Count for each <B> Category, where A is a random quantitative variable and B is a 
        random categorical variable, prints this query in natural language and mongo format, and prints the head(5) of the output

    Parameters
    ----------
    cat_cols : list
        The dataframe that will be ingested to either mongodb or sqlite
    collectionName : str
        Name of the collection we are currently querying on
    """
    random_cat = random.choice(cat_cols)
    nat_language = f"Counts for each {random_cat} category"
    collection = mongo_db[collectionName]
    query_result = collection.aggregate([{"$group": {"_id": f"${random_cat}", "Count": {"$sum": 1}}}, 
                                         {"$project": {f"{random_cat}": "$_id", "Count": 1, "_id": 0}},{"$limit": 5}])
    query = [{"$group": {"_id": f"${random_cat}", "Count": {"$sum": 1}}}, 
                                         {"$project": {f"{random_cat}": "$_id", "Count": 1, "_id": 0}},{"$limit": 5}]
    query_string = ', '.join(json.dumps(part) for part in query)
    data = list(query_result)
    headers = list(data[0].keys())[::-1] if data else []
    # Ensure rows are aligned with reversed headers
    rows = [[row[h] for h in headers] for row in data]
    print(nat_language)
    print(query_string)
    print("Below is the result of the query:")
    print(tabulate(rows, headers=headers, tablefmt="grid"))

def gen_gtlt_query_group(cat_cols, quant_cols, range_, ineq, collectionName):
    """Generates a query in the <B> with Average <A> Greater/Less than X, where A is a random quantitative variable and B is a 
        random categorical variable, and X is a random value within the range of the random quantitative column, and
        prints this query in natural language and mongo format, and prints the head(5) of the output

    Parameters
    ----------
    cat_cols : list
        The dataframe that will be ingested to either mongodb or sqlite
    quant_cols : list
        List of the quantitative variable names
    range_ : dictionary
         Dictionary where the key is the quantitative column and the value is a tuple containing the (min, max) value for said column
    ineq : str
        Name of the ineq that we are using (gt or lt)
    collectionName : str
        Name of the collection we are currently querying on
    """
    if ineq == 'gt':
        ineq_str = "greater"
    elif ineq == 'lt':
         ineq_str = "less"
    else:
        ineq_str = "unknown"
    random_quant = random.choice(quant_cols)
    random_quant_var = f"Average {random_quant}"
    random_cat = random.choice(cat_cols)
    gt_val = random.randint(int(range_[random_quant][0]), int(range_[random_quant][1]))
    nat_language = f"{random_cat} with {random_quant_var} {ineq_str} than {gt_val}"
    collection = mongo_db[collectionName]
    query_result = collection.aggregate([{"$group": {"_id": f"${random_cat}", random_quant_var: {"$avg": f"${random_quant}"}}}, 
                                         {"$match": {random_quant_var: {f"${ineq}": gt_val}}},
                                         {"$project": {f"{random_cat}": "$_id", random_quant_var: 1, "_id": 0}}
                                         ,{"$limit": 5}])
    query = [{"$group": {"_id": f"${random_cat}", random_quant_var: {"$avg": f"${random_quant}"}}}, 
                                         {"$match": {random_quant_var: {f"${ineq}": gt_val}}},
                                         {"$project": {f"{random_cat}": "$_id", random_quant_var: 1, "_id": 0}}
                                         ,{"$limit": 5}]
    query_string = ', '.join(json.dumps(part) for part in query)
    data = list(query_result)
    headers = list(data[0].keys())[::-1] if data else []
    # Ensure rows are aligned with reversed headers
    rows = [[row[h] for h in headers] for row in data]
    print(nat_language)
    print(query_string)
    print("Below is the result of the query:")
    print(tabulate(rows, headers=headers, tablefmt="grid"))

# # Call the download function
download_nltk_resources()
# ingest the json file as a dataframe
filename = "pokedex.json"
df = pd.read_json(f'data/{filename}')
# make sure quantitative variables are in numeric type
numeric, categorical, nested, unique, df_updated = infer_types(df)
range_vals = get_quant_range(df, numeric)
# store updated df in mongodb
collection_name = store_in_mongodb(df_updated, filename)

# test NLP
sample_tokens = process_input("Please provide sample queries")
# generate sample queries
gen_total_query(categorical, numeric, collection_name)
gen_average_query(categorical, numeric, collection_name)
gen_counts_query(categorical, collection_name)
gen_gtlt_query_group(categorical, numeric, range_vals, "gt", collection_name)
gen_gtlt_query_group(categorical, numeric, range_vals, "lt", collection_name)
# generate sample queries with a specific construct, ie. group by
# generate query based on matching template to 