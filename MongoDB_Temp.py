import re
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
import numpy as np

# some learning notes as I have been going:
# don't cover cases where an attribute is a list or dict. These could be queries all on their own. This could be something we add for 
# complexity
# can add a conditional to each generate function to do order by, DESC, ASC, etc. 

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


mongo_client = MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["chat_db"]
mongodb_list = []


# -------------------------------- NLP PORTION --------------------------------
# Download NLTK resources
def download_nltk_resources():
    nltk.download('punkt')  # Required for tokenization
    nltk.download('stopwords')  # For stopword removal
    nltk.download('wordnet')  # For lemmatization

# Define the basic tokenizer (splitting on spaces and lowering text)
def basic_tokenizer(text):
    return text.lower().split()

def process_input_mongo(user_input):
    # Step 1: Tokenize using basic_tokenizer
    tokens = basic_tokenizer(user_input)
    
    # Step 2: Remove stopwords using NLTK stopwords
    # REMOVED THIS SECTION BECAUSE IT GETS RID OF THINGS LIKE BELOW
    # stop_words = set(stopwords.words("english"))
    # tokens = [token for token in tokens if token not in stop_words]
    
    # Step 3: Lemmatize using NLTK WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens
# -------------------------------- NLP PORTION --------------------------------


# -------------------------------- DATA PROCESSING --------------------------------
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
        col.translate(str.maketrans("", "", string.punctuation))
        .replace(" ", "_")
        .lower()
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
# -------------------------------- DATA PROCESSING --------------------------------


# -------------------------------- MONGO MAIN CASES --------------------------------
# main function of getting mongo queries from any sort of user input
def get_mongo_queries_nat(tokens, cat_cols, quant_cols, unique_cols, range_, collectionName):
    query_made = True
    total_tokens = ["sum", "total"]
    average_tokens = ["average", "mean"]
    greater_tokens = ["greater", "more", "than", "above"]
    less_tokens = ["less", "fewer", "below"]
    count_tokens = ["count", "number", "counts"]
    order_tokens = ["ascending", "descending", "order", "sort"]
    cat_chosen = list(set(token.lower() for token in tokens) & set(col.lower() for col in cat_cols))
    quant_chosen = list(set(token.lower() for token in tokens) & set(col.lower() for col in quant_cols))
    unique_chosen = list(set(token.lower() for token in tokens) & set(col.lower() for col in unique_cols))
    extracted_numbers = []
    result = []
    for token in tokens:
        try:
            # Try to convert the token to a number
            num = int(token) if token.isdigit() else float(token)
            extracted_numbers.append(num)  # Stop after finding the first number
        except ValueError:
            continue
    # total case
    if set(tokens) & set(order_tokens):
        order = 'asc' if 'ascending' in tokens else 'desc'
        sort_field = quant_chosen[0] if quant_chosen else (cat_chosen[0] if cat_chosen else None)
        if sort_field:
            result = gen_ordered_query(cat_cols, quant_cols, collectionName, sort_field, order=order)
        
    if set(tokens) & set(total_tokens):
        if(cat_chosen and quant_chosen):
            result = gen_total_query(cat_cols, quant_cols, collectionName, specific_cat=cat_chosen[0], specific_quant=quant_chosen[0])
    elif set(tokens) & set(count_tokens):
        if(cat_chosen):
            result = gen_counts_query(cat_cols, collectionName, specific_cat=cat_chosen[0])
    # average case
    elif set(tokens) & set(average_tokens):
        if(cat_chosen and quant_chosen):
            result = gen_average_query(cat_cols, quant_cols, collectionName, specific_cat=cat_chosen[0], specific_quant=quant_chosen[0])
    # greater than aggregate case
    elif set(tokens) & set(greater_tokens):
        if(cat_chosen and quant_chosen and extracted_numbers):
            result = gen_gtlt_query_group(cat_cols, quant_cols, range_, "gt", collectionName, specific_cat=cat_chosen[0], specific_quant=quant_chosen[0], ineq_input=extracted_numbers[0])
        elif(quant_chosen and extracted_numbers and unique_chosen and not cat_chosen):
            result = gen_gtlt_query_unique(unique_cols, quant_cols, range_, "gt", collectionName, specific_unique=unique_chosen[0], specific_quant=quant_chosen[0], ineq_input=extracted_numbers[0])
    elif set(tokens) & set(less_tokens):
        if(cat_chosen and quant_chosen and extracted_numbers):
            result = gen_gtlt_query_group(cat_cols, quant_cols, range_, "lt", collectionName, specific_cat=cat_chosen[0], specific_quant=quant_chosen[0], ineq_input=extracted_numbers[0])
        elif(quant_chosen and extracted_numbers and unique_chosen and not cat_chosen):
            result = gen_gtlt_query_unique(unique_cols, quant_cols, range_, "lt", collectionName, specific_unique=unique_chosen[0], specific_quant=quant_chosen[0], ineq_input=extracted_numbers[0])
    # do a greater than or less than case
    else:
        print("We couldn't find a query matching your request, please try another and make sure you use the correct column names in your queries")
    return result

# get general sample queries
def get_sample_mongo_gen(cat_cols, quant_cols, unique_cols, range_, collectionName):
    all_queries = {}
    # return values for these are: return [query_result, query_string, nat_language]
    all_queries['Total'] = gen_total_query(cat_cols, quant_cols, collectionName)
    all_queries['Average'] = gen_average_query(cat_cols, quant_cols, collectionName)
    all_queries['Counts'] = gen_counts_query(cat_cols, collectionName)
    all_queries['Greater_group'] = gen_gtlt_query_group(cat_cols, quant_cols, range_, 'gt', collectionName)
    all_queries['Less_group'] = gen_gtlt_query_group(cat_cols, quant_cols, range_, 'lt', collectionName)
    if unique_cols:
        all_queries['Greater_unique'] = gen_gtlt_query_unique(unique_cols, quant_cols, range_, 'gt', collectionName)
        all_queries['Less_unique'] = gen_gtlt_query_unique(unique_cols, quant_cols, range_, 'lt', collectionName)

    # Use list() to convert dictionary items into a sequence
    selected_pairs = random.sample(list(all_queries.items()), 3)

    # Convert back to dictionary if needed
    selected_dict = dict(selected_pairs)

    return selected_dict

# get specific sample queries (ie group by)
def get_sample_mongo_specific(tokens, cat_cols, quant_cols, unique_cols, range_, collectionName):
    aggregate_tokens = ['aggregate', 'group']
    find_tokens = ['find', 'where', 'retrieve']
    # if its aggregate:
    if set(tokens) & set(aggregate_tokens):
        all_queries = {}
        all_queries['Total'] = gen_total_query(cat_cols, quant_cols, collectionName)
        all_queries['Average'] = gen_average_query(cat_cols, quant_cols, collectionName)
        all_queries['Counts'] = gen_counts_query(cat_cols, collectionName)
        all_queries['Greater_group'] = gen_gtlt_query_group(cat_cols, quant_cols, range_, 'gt', collectionName)
        all_queries['Less_group'] = gen_gtlt_query_group(cat_cols, quant_cols, range_, 'lt', collectionName)
        selected_pairs = random.sample(list(all_queries.items()), 1)
        selected_value = selected_pairs[0][1]
        data = selected_value
        return data
    elif set(tokens) & set(find_tokens):
        all_queries = {}
        all_queries['Greater_unique'] = gen_gtlt_query_unique(unique_cols, quant_cols, range_, 'gt', collectionName)
        all_queries['Less_unique'] = gen_gtlt_query_unique(unique_cols, quant_cols, range_, 'lt', collectionName)
        
        selected_pairs = random.sample(list(all_queries.items()), 1)
        selected_value = selected_pairs[0][1]
        data = selected_value
        return data
    else:
        print("We couldn't quite find the query type you were looking for. Here are some suggestions: \n- aggregate\n- find")

# -------------------------------- MONGO MAIN CASES --------------------------------

# -------------------------------- MONGO HELPER FUNCTIONS --------------------------------
def gen_ordered_query(cat_cols, quant_cols, collectionName, sort_field, order='asc', specific_cat=None, specific_quant=None):
    """Generates a query with ordering (ascending/descending) based on a field

    Parameters
    ----------
    cat_cols : list
        List of categorical variable names.
    quant_cols : list
        List of quantitative variable names.
    collectionName : str
        Name of the MongoDB collection.
    sort_field : str
        Field to sort by.
    order : str
        Sorting order, either 'asc' for ascending or 'desc' for descending.
    specific_cat : str
        Optional specific categorical variable to use.
    specific_quant : str
        Optional specific quantitative variable to use.

    Returns
    -------
    list
        Query result, query string, and natural language description.
    """
    order_val = 1 if order == 'asc' else -1
    random_cat = specific_cat if specific_cat else random.choice(cat_cols)
    random_quant = specific_quant if specific_quant else random.choice(quant_cols)
    
    nat_language = f"Order data by {sort_field} in {'ascending' if order_val == 1 else 'descending'} order"
    collection = mongo_db[collectionName]
    query_result = collection.aggregate([
        {"$group": {"_id": f"${random_cat}", random_quant: {"$sum": f"${random_quant}"}}},
        {"$sort": {sort_field: order_val}},
        {"$project": {f"{random_cat}": "$_id", random_quant: 1, "_id": 0}},
        {"$limit": 5}
    ])
    query = [
        {"$group": {"_id": f"${random_cat}", random_quant: {"$sum": f"${random_quant}"}}},
        {"$sort": {sort_field: order_val}},
        {"$project": {f"{random_cat}": "$_id", random_quant: 1, "_id": 0}},
        {"$limit": 5}
    ]
    query_string = f"collection.aggregate({json.dumps(query)})"
    return [query_result, query_string, nat_language, "Order"]

def gen_total_query(cat_cols, quant_cols, collectionName, specific_cat=None, specific_quant=None):
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
    random_quant = specific_quant if specific_quant else random.choice(quant_cols)
    random_cat = specific_cat if specific_cat else random.choice(cat_cols)
    random_quant_var = f"Total {random_quant}"
    random_quant_var = random_quant_var.translate(str.maketrans('', '',
                                    string.punctuation))
    nat_language = f"Total {random_quant} for each {random_cat} category"
    collection = mongo_db[collectionName]
    query_result = collection.aggregate([{"$group": {"_id": f"${random_cat}", random_quant_var: {"$sum": f"${random_quant}"}}}, 
                                         {"$project": {f"{random_cat}": "$_id", random_quant_var: 1, "_id": 0}},{"$limit": 5}])
    query = [{"$group": {"_id": f"${random_cat}", random_quant_var: {"$sum": f"${random_quant}"}}}, 
                                         {"$project": {f"{random_cat}": "$_id", random_quant_var: 1, "_id": 0}},{"$limit": 5}]
    query_string = f"collection.aggregate({json.dumps(query)})"
    return [query_result, query_string, nat_language, "Total"]
    
def gen_average_query(cat_cols, quant_cols, collectionName, specific_cat=None, specific_quant=None):
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
    random_quant = specific_quant if specific_quant else random.choice(quant_cols)
    random_cat = specific_cat if specific_cat else random.choice(cat_cols)
    random_quant_var = f"Average {random_quant}"
    random_quant_var = random_quant_var.translate(str.maketrans('', '',
                                    string.punctuation))
    nat_language = f"Average {random_quant} for each {random_cat} category"
    collection = mongo_db[collectionName]
    query_result = collection.aggregate([{"$group": {"_id": f"${random_cat}", random_quant_var: {"$avg": f"${random_quant}"}}}, 
                                         {"$project": {f"{random_cat}": "$_id", random_quant_var: 1, "_id": 0}},{"$limit": 5}])
    query = [{"$group": {"_id": f"${random_cat}", random_quant_var: {"$avg": f"${random_quant}"}}}, 
                                         {"$project": {f"{random_cat}": "$_id", random_quant_var: 1, "_id": 0}},{"$limit": 5}]
    query_string = f"collection.aggregate({json.dumps(query)})"
    return [query_result, query_string, nat_language, "Average"]

def gen_counts_query(cat_cols, collectionName, specific_cat=None):
    """Generates a query in the Count for each <B> Category, where A is a random quantitative variable and B is a 
        random categorical variable, prints this query in natural language and mongo format, and prints the head(5) of the output

    Parameters
    ----------
    cat_cols : list
        The dataframe that will be ingested to either mongodb or sqlite
    collectionName : str
        Name of the collection we are currently querying on
    """
    random_cat = specific_cat if specific_cat else random.choice(cat_cols)
    nat_language = f"Counts for each {random_cat} category"
    collection = mongo_db[collectionName]
    query_result = collection.aggregate([{"$group": {"_id": f"${random_cat}", "Count": {"$sum": 1}}}, 
                                         {"$project": {f"{random_cat}": "$_id", "Count": 1, "_id": 0}},{"$limit": 5}])
    query = [{"$group": {"_id": f"${random_cat}", "Count": {"$sum": 1}}}, 
                                         {"$project": {f"{random_cat}": "$_id", "Count": 1, "_id": 0}},{"$limit": 5}]
    query_string = f"collection.aggregate({json.dumps(query)})"
    return [query_result, query_string, nat_language, "Counts"]

def gen_gtlt_query_group(cat_cols, quant_cols, range_, ineq, collectionName, specific_cat=None, specific_quant=None, ineq_input=None):
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
    random_quant = specific_quant if specific_quant else random.choice(quant_cols)
    random_quant_var = f"Average {random_quant}"
    random_cat = specific_cat if specific_cat else random.choice(cat_cols)
    ineq_val = ineq_input if ineq_input else random.randint(int(range_[random_quant][0]) + 1, int(range_[random_quant][1]) - 1)
    nat_language = f"{random_cat} with {random_quant_var} {ineq_str} than {ineq_val}"
    collection = mongo_db[collectionName]
    query_result = collection.aggregate([{"$group": {"_id": f"${random_cat}", random_quant_var: {"$avg": f"${random_quant}"}}}, 
                                         {"$match": {random_quant_var: {f"${ineq}": ineq_val}}},
                                         {"$project": {f"{random_cat}": "$_id", random_quant_var: 1, "_id": 0}}
                                         ,{"$limit": 5}])
    query = [{"$group": {"_id": f"${random_cat}", random_quant_var: {"$avg": f"${random_quant}"}}}, 
                                         {"$match": {random_quant_var: {f"${ineq}": ineq_val}}},
                                         {"$project": {f"{random_cat}": "$_id", random_quant_var: 1, "_id": 0}}
                                         ,{"$limit": 5}]
    query_string = f"collection.aggregate({json.dumps(query)})"
    return [query_result, query_string, nat_language, ineq_str + " than"]

def gen_gtlt_query_unique(unique_cols, quant_cols, range_, ineq, collectionName, specific_unique=None, specific_quant=None, ineq_input=None):
    """Generates a query <B> with  <A> Greater/Less than X, where A is a random quantitative variable and B is a 
        random categorical variable with unique values, and X is a random value within the range of the random quantitative column, and
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
    random_quant = specific_quant if specific_quant else random.choice(quant_cols)
    random_unique = specific_unique if specific_unique else random.choice(unique_cols)
    ineq_val = ineq_input if ineq_input else random.randint(int(range_[random_quant][0]), int(range_[random_quant][1]))
    nat_language = f"Find {random_unique} with {random_quant} {ineq_str} than {ineq_val}"
    collection = mongo_db[collectionName]
    query_result = collection.find({f"{random_quant}": {f"${ineq}": ineq_val}},
                                   {f"{random_unique}": 1, f"{random_quant}": 1, "_id": 0}).limit(5)
    query = [{f"{random_quant}": {f"${ineq}": ineq_val}}, {f"{random_unique}": 1, f"{random_quant}": 1, "_id": 0}]
    query_string = f"collection.find({json.dumps(query)})"
    return [query_result, query_string, nat_language, ineq_str + " than"]
# -------------------------------- MONGO HELPER FUNCTIONS --------------------------------


# setup
# download_nltk_resources()
# filename = "used_car.json"
# df = pd.read_json(f'data/{filename}')
# # # preprocessing
# numeric, categorical, nested, unique, df_updated = infer_types(df)
# print("NUMERIC BASE FILE", numeric)
# print("NUMERIC", numeric)
# print("CATEGORICAL", categorical)
# print("UNIQUE", unique)
# range_vals = get_quant_range(df, numeric)
# collection_name = store_in_mongodb(df_updated, filename)

# # TEST REQ 1: GET EXAMPLE QUERIES USING TEMPLATE
# get_sample_mongo_gen(categorical, numeric, unique, range_vals, collection_name)

# TEST REQ 2: GET EXAMPLE QUERY(S) USING TEMPLATE WITH SPECIFIC LANGUAGE CONSTRUCTS
# user_query = "Please provide sample queries using aggregate"
# user_query2 = "Please provide a sample query using find"
# user_query3 = "Please provide a sample query using meow"
# tokens_agg = process_input(user_query)
# tokens_find = process_input(user_query2)
# tokens_wrong = process_input(user_query3)
# get_sample_mongo_specific(tokens_agg, categorical, numeric, unique, range_vals, collection_name)
# get_sample_mongo_specific(tokens_find, categorical, numeric, unique, range_vals, collection_name)
# get_sample_mongo_specific(tokens_wrong, categorical, numeric, unique, range_vals, collection_name)

# TEST REQ 3: GET QUERY FROM NATURAL LANGUAGE
# input = "Counts for pokemon in type_1 category"
# tokens = process_input(input)
# print(tokens)
# get_mongo_queries_nat(tokens, categorical, numeric, unique, range_vals, collection_name)


# NEXT STEP, COMBINING QUERIES!!!!!
