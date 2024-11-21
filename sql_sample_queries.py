import random

# Categorizes columns to be quantitative or qualitative so that we can map them to query templates
def categorize_columns(dataframe):
    categorical = [col for col in dataframe.columns if dataframe[col].dtype == 'object']
    quantitative = [col for col in dataframe.columns if dataframe[col].dtype in ['int64', 'float64']]
    return categorical, quantitative

templates = [
    "SELECT {column}, SUM({value_column}) AS total_{value_column} FROM {table} GROUP BY {column}",
    "SELECT {column}, AVG({value_column}) AS avg_{value_column} FROM {table} GROUP BY {column}",
    "SELECT {column}, MAX({value_column}) AS max_{value_column} FROM {table} GROUP BY {column}",
    "SELECT COUNT(*) AS total_records FROM {table} WHERE {column} = 'some_value'"
]

# Generates sample queries
def generate_sample_queries(table_name, categorical, quantitative):
    queries = []
    for _ in range(3):  # Generate 3 sample queries
        template = random.choice(templates)
        column = random.choice(categorical)
        value_column = random.choice(quantitative)
        queries.append(template.format(column=column, value_column=value_column, table=table_name))
    return queries


def generate_construct_queries(construct, table_name, categorical, quantitative):
    if construct == "group by":
        return [
            f"SELECT {col}, COUNT(*) AS count_{col} FROM {table_name} GROUP BY {col}"
            for col in categorical
        ]
    elif construct == "where":
        return [
            f"SELECT {col} FROM {table_name} WHERE {col} = 'some_value'"
            for col in categorical
        ]
    # Add more constructs as needed
    return []
