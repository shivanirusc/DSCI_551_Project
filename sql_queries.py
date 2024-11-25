import random
import streamlit as st
import pandas as pd
import sqlite3

# Categorizes columns to be quantitative or qualitative so that we can map them to query templates
def categorize_columns(dataframe):
    categorical = [col for col in dataframe.columns if dataframe[col].dtype == 'object']
    quantitative = [col for col in dataframe.columns if dataframe[col].dtype in ['int64', 'float64']]
    return categorical, quantitative

# Query templates that will get used 
templates = [
    # Group By with Aggregation
    (
        "SELECT {column}, SUM({value_column}) AS total_{value_column} FROM {table} GROUP BY {column}",
        "Total {value_column} by {column}:"
    ),
    # Group By with Aggregation and Condition in WHERE
    (
        "SELECT {column}, AVG({value_column}) AS avg_{value_column} FROM {table} WHERE {value_column} != 70 GROUP BY {column}",
        "Average {value_column} by {column}, excluding rows where {value_column} equals 70:"
    ),
    # Group By with Maximum Value
    (
        "SELECT {column}, MAX({value_column}) AS max_{value_column} FROM {table} GROUP BY {column}",
        "Maximum {value_column} by {column}:"
    ),
    # Group By with Aggregation and HAVING Clause
    (
        "SELECT {column}, AVG({value_column}) AS avg_{value_column} FROM {table} GROUP BY {column} HAVING avg_{value_column} > 50",
        "Average {value_column} by {column}, where the average is greater than 50:"
    ),
    # Order By in Ascending Order
    (
        "SELECT {column}, {value_column} FROM {table} ORDER BY {value_column} ASC",
        "{value_column} for each {column}, ordered in ascending order of {value_column}:"
    ),
    # Order By in Descending Order with Limit
    (
        "SELECT {column}, {value_column} FROM {table} ORDER BY {value_column} DESC LIMIT 12",
        "Top 12 {column} values ordered by {value_column} in descending order:"
    )
]

# Generates sample queries
def generate_sample_queries(table_name, categorical, quantitative):
    if not categorical or not quantitative:
        # Handle cases where the dataset doesn't meet requirements
        return 0
    
    queries = []
    for _ in range(3):  # Generate 3 sample queries
        sql, nl = random.choice(templates)
        column = random.choice(categorical)
        value_column = random.choice(quantitative)

        sql_query = sql.format(column=column, value_column=value_column, table=table_name)
        nl_query = nl.format(column=column, value_column=value_column)

        queries.append((sql_query, nl_query))   

    return queries

# Query Execution
def execute_query(query):
    conn = sqlite3.connect("chatdb_sql.db")
    result = pd.read_sql_query(query, conn)
    st.write("**Query Result from SQLite:**")
    st.dataframe(result)

# Will generate query for following language constructs: group by, having, order by, aggregation, where
def generate_construct_queries(construct, table_name, categorical, quantitative):
    if not categorical or not quantitative:
        # Handle cases where the dataset doesn't meet requirements
        return 0
    
    queries_with_nl = []

    if construct == "group by":
        # Group by queries
        templates = [
            (
                "SELECT {column}, COUNT(*) AS count_{column} FROM {table} GROUP BY {column}",
                "Count of {column} grouped by {column}."
            ),
            (
                "SELECT {column}, SUM({value_column}) AS total_{value_column} FROM {table} GROUP BY {column}",
                "Total {value_column} by {column}."
            ),
            (
                "SELECT {column}, MAX({value_column}) AS max_{value_column} FROM {table} GROUP BY {column}",
                "Maximum {value_column} by {column}."
            ),
        ]
    elif construct == "where":
        # Where queries with a condition
        templates = [
            (
                "SELECT {column}, {value_column} FROM {table} WHERE {value_column} > 50",
                "{value_column} for each {column} where {value_column} is greater than 50."
            ),
            (
                "SELECT {column}, {value_column} FROM {table} WHERE {value_column} > 80",
                "{value_column} for each {column} where {value_column} is greater than 80."
            ),
            (
                "SELECT {column}, {value_column} FROM {table} WHERE {value_column} < 100",
                "{value_column} for each {column} where {value_column} is less than 100."
            )
        ]
    elif construct == "having":
        # Having clause queries with aggregation
        templates = [
            (
                "SELECT {column}, SUM({value_column}) AS total_{value_column} FROM {table} GROUP BY {column} HAVING total_{value_column} > 500",
                "Total {value_column} grouped by {column}, where the total is greater than 500."
            ),
            (
                "SELECT {column}, SUM({value_column}) AS total_{value_column} FROM {table} GROUP BY {column} HAVING total_{value_column} > 150",
                "Total {value_column} grouped by {column}, where the total is greater than 150."
            ),
            (
                "SELECT {column}, SUM({value_column}) AS total_{value_column} FROM {table} GROUP BY {column} HAVING total_{value_column} < 500",
                "Total {value_column} grouped by {column}, where the total is less than 500."
            ),
            (
                "SELECT {column}, SUM({value_column}) AS total_{value_column} FROM {table} GROUP BY {column} HAVING total_{value_column} < 150",
                "Total {value_column} grouped by {column}, where the total is less than 150."
            ),
        ]
    elif construct == "order by":
        # Order by queries
        templates = [
            (
                "SELECT {column}, {value_column} FROM {table} ORDER BY {value_column} DESC LIMIT 10",
                "Top 10 {column} values ordered by {value_column} in descending order."
            ),
            (
                "SELECT {column}, {value_column} FROM {table} ORDER BY {value_column} ASC",
                "{value_column} for each {column}, ordered in ascending order of {value_column}."
            ),
            (
                "SELECT {column}, {value_column} FROM {table} ORDER BY {value_column} DESC LIMIT 50",
                "Top 50 {column} values ordered by {value_column} in descending order."
            ),
            (
                "SELECT {column}, {value_column} FROM {table} ORDER BY {value_column} ASC LIMIT 10",
                "Top 10 {value_column} for each {column}, ordered in ascending order of {value_column}."
            )
        ]
    elif construct == "aggregation":
        # Aggregation queries
        templates = [
            (
                "SELECT SUM({value_column}) AS total_{value_column}, AVG({value_column}) AS avg_{value_column} FROM {table}",
                "Total and average of {value_column} across all records in {table}."
            ),
            (
                "SELECT {column}, SUM({value_column}) AS total_{value_column} FROM {table} GROUP BY {column}",
                "Total {value_column} by {column}."
            ),
            (
                "SELECT {column}, AVG({value_column}) AS avg_{value_column} FROM {table} GROUP BY {column}",
                "Average {value_column} by {column}."
            ),
        ]
    else:
        return 0

    # Generate 3 random queries based on the construct
    for _ in range(3):
        sql, nl = random.choice(templates)
        column = random.choice(categorical)
        value_column = random.choice(quantitative)

        sql_query = sql.format(column=column, value_column=value_column, table=table_name)
        nl_query = nl.format(column=column, value_column=value_column, table=table_name)

        queries_with_nl.append((sql_query, nl_query))
    
    return queries_with_nl