# -*- coding: utf-8 -*-
"""DSCI551ProjectSQL.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/12CnlUC2nm9G18xQ9brj6cDv5EsghSH7k
"""

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import streamlit as st

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

# SQL query templates
sql_queries = {
    "monthly_sales_2022": """
        SELECT strftime('%Y-%m', Order_Date) AS Month, SUM(Sales) AS Total_Sales
        FROM furniture_sales
        WHERE strftime('%Y', Order_Date) = '2022'
        GROUP BY Month
        ORDER BY Month;
    """,
    "total_sales_by_year": """
        SELECT strftime('%Y', Order_Date) AS Year, SUM(Sales) AS Total_Sales
        FROM furniture_sales
        GROUP BY Year
        ORDER BY Year;
    """,
    "top_5_cities_highest_sales": """
        SELECT City, SUM(Sales) AS Total_Sales
        FROM furniture_sales
        GROUP BY City
        ORDER BY Total_Sales DESC
        LIMIT 5;
    """,
    "first_10_rows": """
        SELECT *
        FROM furniture_sales
        LIMIT 10;
    """,
    "unique_product_categories": """
        SELECT DISTINCT Category
        FROM furniture_sales;
    """,
    "orders_with_profit_above_100": """
        SELECT *
        FROM furniture_sales
        WHERE Profit > 100;
    """,
    "orders_shipped_first_class": """
        SELECT *
        FROM furniture_sales
        WHERE Ship_Mode = 'First Class';
    """,
    "total_sales_by_category": """
        SELECT Category, SUM(Sales) AS Total_Sales
        FROM furniture_sales
        GROUP BY Category;
    """,
    "profit_discount_by_region": """
        SELECT Region, SUM(Profit) AS Total_Profit, AVG(Discount) AS Average_Discount
        FROM furniture_sales
        GROUP BY Region;
    """,
    "top_5_profitable_products": """
        SELECT Product_Name, SUM(Profit) AS Total_Profit
        FROM furniture_sales
        GROUP BY Product_Name
        ORDER BY Total_Profit DESC
        LIMIT 5;
    """,
    "office_supplies_above_20_discount": """
        SELECT *
        FROM furniture_sales
        WHERE Category = 'Office Supplies' AND Discount > 0.20;
    """,
    "total_sales_by_state": """
        SELECT State, SUM(Sales) AS Total_Sales
        FROM furniture_sales
        GROUP BY State
        ORDER BY Total_Sales DESC;
    """,
    "customer_total_purchases": """
        SELECT Customer_Name, SUM(Sales) AS Total_Purchase
        FROM furniture_sales
        GROUP BY Customer_Name
        ORDER BY Total_Purchase DESC;
    """,
}

# Keywords for matching
query_keywords = {
    "monthly_sales_2022": ["monthly", "sales", "2022"],
    "total_sales_by_year": ["total", "sales", "year"],
    "top_5_cities_highest_sales": ["top", "5", "cities", "highest", "sales"],
    "first_10_rows": ["first", "10", "rows"],
    "unique_product_categories": ["unique", "product", "categories"],
    "orders_with_profit_above_100": ["orders", "profit", "above", "100"],
    "orders_shipped_first_class": ["orders", "shipped", "first", "class"],
    "total_sales_by_category": ["total", "sales", "category"],
    "profit_discount_by_region": ["profit", "discount", "region"],
    "top_5_profitable_products": ["top", "5", "profitable", "products"],
    "office_supplies_above_20_discount": ["office", "supplies", "discount", "above", "20"],
    "total_sales_by_state": ["total", "sales", "state"],
    "customer_total_purchases": ["customer", "total", "purchases"],
}

# Initialize stemmer and stop words
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    """Tokenize, remove stopwords, and stem words."""
    tokens = word_tokenize(text.lower())
    filtered_tokens = [stemmer.stem(w) for w in tokens if w.isalnum() and w not in stop_words]
    return filtered_tokens

def find_best_match(user_input, query_keywords):
    """Find the best SQL query template based on keyword matches."""
    processed_input = preprocess_text(user_input)
    best_match = None
    highest_score = 0

    for query, keywords in query_keywords.items():
        # Calculate match score based on keyword overlap
        score = sum(1 for word in processed_input if word in keywords)

        if score > highest_score:
            highest_score = score
            best_match = query

    return best_match

# Streamlit app setup
st.title("ChatDB: Interactive SQL Query Generator with NLTK")

# User input
user_question = st.text_input("Ask a question about the furniture sales dataset:")

# Process the question and display the SQL query
if user_question:
    best_match = find_best_match(user_question, query_keywords)

    if best_match:
        matching_query = sql_queries[best_match]
        st.write("Generated SQL Query:")
        st.code(matching_query, language='sql')
    else:
        st.write("Sorry, I couldn't find a matching SQL query for that question.")