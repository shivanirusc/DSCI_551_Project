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
    # Clear old data before inserting new
    collection.drop()
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
        
        # Calculate the buffer size (5% of the range by default)
        range_span = max_val - min_val
        buffer = range_span * 0.10
        
        # Ensure that the range includes the data, add the buffer
        buffered_min = min_val - buffer if min_val - buffer >= 0 else 0  # Avoid negative values
        buffered_max = max_val + buffer
        
        range_dict[col] = (buffered_min, buffered_max)
    return range_dict
# -------------------------------- DATA PROCESSING --------------------------------


# -------------------------------- MONGO MAIN CASES --------------------------------
# main function of getting mongo queries from any sort of user input
def get_mongo_queries_nat(user_input, tokens, cat_cols, quant_cols, unique_cols, range_, collectionName):
    total_tokens = ["sum", "total"]
    average_tokens = ["average", "mean"]
    greater_tokens = ["greater", "more", "above"]
    less_tokens = ["less", "fewer", "below"]
    count_tokens = ["count", "number", "counts"]
    order_tokens = ["ascending", "descending", "order", "sort"]
    cat_chosen = list(set(token.lower() for token in tokens) & set(col.lower() for col in cat_cols))
    quant_chosen = list(set(token.lower() for token in tokens) & set(col.lower() for col in quant_cols))
    unique_chosen = list(set(token.lower() for token in tokens) & set(col.lower() for col in unique_cols))
    extracted_numbers = []
    result = []
    user_input = user_input.lower()
    tokens_sort = user_input.split()
    sort_field_ = ""
    if "sort" in tokens_sort and "by" in tokens_sort:
        sort_index = tokens_sort.index("sort")
        if sort_index + 2 < len(tokens_sort):
            sort_field_ = tokens_sort[sort_index + 2]
            if sort_field_ == "total" and  sort_index + 3 < len(tokens_sort):
                sort_field_ = sort_field_.capitalize() + " " + tokens_sort[sort_index + 3]
            if sort_field_ == "average" and  sort_index + 3 < len(tokens_sort):
                sort_field_ = sort_field_.capitalize() + " " + tokens_sort[sort_index + 3]
    sort_order_ = "asc"
    if "descending" in user_input:
        sort_order_ = "desc"
    for token in tokens:
        try:
            # Try to convert the token to a number
            num = int(token) if token.isdigit() else float(token)
            extracted_numbers.append(num)  # Stop after finding the first number
        except ValueError:
            continue
    # total case
    if set(tokens) & set(total_tokens):
        if len(quant_chosen) > 1 and 'total' in quant_chosen:
            quant_chosen.remove('total')
        print(quant_chosen)
        if(cat_chosen and quant_chosen):
            if "sort" in tokens or "order" in tokens:
                if sort_field_:
                    result = gen_total_query(cat_cols, quant_cols, collectionName, specific_cat=cat_chosen[0], specific_quant=quant_chosen[0], sort_field=sort_field_, order=sort_order_)
                else:
                    result = gen_total_query(cat_cols, quant_cols, collectionName, specific_cat=cat_chosen[0], specific_quant=quant_chosen[0], sort_field=quant_chosen[0], order=sort_order_)
            else:
                result = gen_total_query(cat_cols, quant_cols, collectionName, specific_cat=cat_chosen[0], specific_quant=quant_chosen[0])
    elif set(tokens) & set(greater_tokens):
        if(cat_chosen and quant_chosen and extracted_numbers):
            result = gen_gtlt_query_group(cat_cols, quant_cols, range_, "gt", collectionName, specific_cat=cat_chosen[0], specific_quant=quant_chosen[0], ineq_input=extracted_numbers[0])
        elif(quant_chosen and extracted_numbers and unique_chosen and not cat_chosen):
            result = gen_gtlt_query_unique(unique_cols, quant_cols, range_, "gt", collectionName, specific_unique=unique_chosen[0], specific_quant=quant_chosen[0], ineq_input=extracted_numbers[0])
    elif set(tokens) & set(count_tokens):
        if(cat_chosen):
            if "sort" in tokens or "order" in tokens:
                if sort_field_:
                    result = gen_counts_query(cat_cols, collectionName, specific_cat=cat_chosen[0], sort_field = sort_field_.capitalize(), order=sort_order_)
                else:
                    result = gen_counts_query(cat_cols, collectionName, specific_cat=cat_chosen[0], sort_field="Count", order=sort_order_)
            else:
                result = gen_counts_query(cat_cols, collectionName, specific_cat=cat_chosen[0])
    # less than aggregate case
    elif set(tokens) & set(less_tokens):
        if(cat_chosen and quant_chosen and extracted_numbers):
            result = gen_gtlt_query_group(cat_cols, quant_cols, range_, "lt", collectionName, specific_cat=cat_chosen[0], specific_quant=quant_chosen[0], ineq_input=extracted_numbers[0])
        elif(quant_chosen and extracted_numbers and unique_chosen and not cat_chosen):
            result = gen_gtlt_query_unique(unique_cols, quant_cols, range_, "lt", collectionName, specific_unique=unique_chosen[0], specific_quant=quant_chosen[0], ineq_input=extracted_numbers[0])
    # average case
    elif set(tokens) & set(average_tokens):
        if(cat_chosen and quant_chosen):
            if "sort" in tokens or "order" in tokens:
                if sort_field_:
                    result = gen_average_query(cat_cols, quant_cols, collectionName, specific_cat=cat_chosen[0], specific_quant=quant_chosen[0],  sort_field=sort_field_, order=sort_order_)
                else:
                    result = gen_average_query(cat_cols, quant_cols, collectionName, specific_cat=cat_chosen[0], specific_quant=quant_chosen[0],  sort_field=quant_chosen[0], order=sort_order_)
            else:
                result = gen_average_query(cat_cols, quant_cols, collectionName, specific_cat=cat_chosen[0], specific_quant=quant_chosen[0])
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
    sort_tokens = ['sort', 'order']
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
    elif set(tokens) & set(sort_tokens):
        all_queries = {}
        cat_col = random.choice(cat_cols)
        quant_col = random.choice(quant_cols)
        sort_total = "Total " + quant_col
        sort_average = "Average " + quant_col
        order = ["asc", "desc"]
        order_chosen = random.choice(order)
        all_queries['Total'] = gen_total_query(cat_cols, quant_cols, collectionName, specific_cat = cat_col, specific_quant = quant_col, sort_field=sort_total, order = order_chosen)
        all_queries['Average'] = gen_average_query(cat_cols, quant_cols, collectionName, specific_quant = quant_col, sort_field=sort_average, order = order_chosen)
        selected_pairs = random.sample(list(all_queries.items()), 1)
        selected_value = selected_pairs[0][1]
        data = selected_value
        return data
    else:
        print("We couldn't quite find the query type you were looking for. Here are some suggestions: \n- aggregate\n- find")

# -------------------------------- MONGO MAIN CASES --------------------------------

# -------------------------------- MONGO HELPER FUNCTIONS --------------------------------
def gen_total_query(cat_cols, quant_cols, collectionName, specific_cat=None, specific_quant=None, sort_field=None, order='asc', limit=5):
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
    order_val = 1 if order == 'asc' else -1
    random_quant = specific_quant if specific_quant else random.choice(quant_cols)
    random_cat = specific_cat if specific_cat else random.choice(cat_cols)
    random_quant_var = f"Total {random_quant}"
    random_quant_var = random_quant_var.translate(str.maketrans('', '',
                                    string.punctuation))
    nat_language = f"Total {random_quant} for each {random_cat} category"
    collection = mongo_db[collectionName]
    pipeline = [{"$group": {"_id": f"${random_cat}", f"Total {random_quant}": {"$sum": f"${random_quant}"}}}, 
                                         {"$project": {f"{random_cat}": "$_id", f"Total {random_quant}": 1, "_id": 0}}]
    if sort_field:
        pipeline.append({"$sort": {sort_field: order_val}})
        nat_language = nat_language + " sorted by " + sort_field + " in " + order + " order"
    pipeline.append({"$limit": limit})
    query_result = collection.aggregate(pipeline)
    query_string = f"collection.aggregate({json.dumps(pipeline)})"
    return [query_result, query_string, nat_language, "Total"]
    
def gen_average_query(cat_cols, quant_cols, collectionName, specific_cat=None, specific_quant=None,  sort_field=None, order='asc', limit=5):
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
    order_val = 1 if order == 'asc' else -1
    random_quant = specific_quant if specific_quant else random.choice(quant_cols)
    random_cat = specific_cat if specific_cat else random.choice(cat_cols)
    random_quant_var = f"Average {random_quant}"
    random_quant_var = random_quant_var.translate(str.maketrans('', '',
                                    string.punctuation))
    nat_language = f"Average {random_quant} for each {random_cat} category"
    collection = mongo_db[collectionName]
    pipeline = [{"$group": {"_id": f"${random_cat}", f"Average {random_quant}": {"$avg": f"${random_quant}"}}}, 
                                         {"$project": {f"{random_cat}": "$_id", f"Average {random_quant}": 1, "_id": 0}}]
    if sort_field:
        pipeline.append({"$sort": {sort_field: order_val}})
        nat_language = nat_language + " sorted by " + sort_field + " in " + order + " order"
    pipeline.append({"$limit": limit})
    query_result = collection.aggregate(pipeline)
    query_string = f"collection.aggregate({json.dumps(pipeline)})"
    return [query_result, query_string, nat_language, "Average"]

def gen_counts_query(cat_cols, collectionName, specific_cat=None,  sort_field=None, order='asc', limit=5):
    """Generates a query in the Count for each <B> Category, where A is a random quantitative variable and B is a 
        random categorical variable, prints this query in natural language and mongo format, and prints the head(5) of the output

    Parameters
    ----------
    cat_cols : list
        The dataframe that will be ingested to either mongodb or sqlite
    collectionName : str
        Name of the collection we are currently querying on
    """
    order_val = 1 if order == 'asc' else -1
    random_cat = specific_cat if specific_cat else random.choice(cat_cols)
    nat_language = f"Counts for each {random_cat} category"
    collection = mongo_db[collectionName]
    pipeline = [{"$group": {"_id": f"${random_cat}", "Count": {"$sum": 1}}}, 
                                         {"$project": {f"{random_cat}": "$_id", "Count": 1, "_id": 0}}]
    if sort_field:
        pipeline.append({"$sort": {sort_field: order_val}})
        nat_language = nat_language + " sorted by " + sort_field + " in " + order + " order"
    pipeline.append({"$limit": limit})
    query_result = collection.aggregate(pipeline)
    query_string = f"collection.aggregate({json.dumps(pipeline)})"
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
    query_result = collection.aggregate([{"$group": {"_id": f"${random_cat}", f"Average {random_quant}": {"$avg": f"${random_quant}"}}}, 
                                         {"$match": {random_quant_var: {f"${ineq}": ineq_val}}},
                                         {"$project": {f"{random_cat}": "$_id", f"Average {random_quant}": 1, "_id": 0}}
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