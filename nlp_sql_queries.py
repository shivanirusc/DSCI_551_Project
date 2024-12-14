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
from streamlit_extras.stylable_container import stylable_container

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
                conditions.append(token.upper())
        
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
