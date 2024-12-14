# ChatDB

## master_chatdb.py is the main application file for interacting with datasets through natural language queries. It uses NLP to convert your input into SQL or MongoDB queries, returning results instantly.

**Quick Start**
**Clone & Install:**

>> git clone https://github.com/shivanirusc/DSCI_551_Project.git

>> cd DSCI_551_Project

>> pip install -r requirements.txt

**Run the App:**

>>streamlit run master_chatdb.py

Open the local URL (http://localhost:27017).

**Use the Interface:**

Upload a CSV (for SQL) or JSON (for MongoDB).

Ask questions in plain English (e.g., “Show total sales by region”).

View queries and results directly in the browser.

**Direct Link**

Click here to open the hosted Streamlit app: https://dsci551project-ggthrcj6wsvlvoch2d2lvd.streamlit.app/

## Example Inputs
- MySQL
    - Furniture Dataset:
        1. example sql query
        2. example sql query with group by
        3. Find the total sales amount by product category
        4. What is the average discount percentage by brand?
        5. Find products where inventory is less than 50 and (sales are greater than 100 or revenue is greater than 500).

    - Cars Dataset:
        1. What is the average price of cars by car ID?
        2. example sql query
        3. example sql query with aggregation


- MongoDB
    - Pokemon dataset:
        1. example mongodb queries
        2. example mongodb queries using aggregate
        3. example mongodb queries using find
        4. Please give total hp by type_1 and sort by total hp in descending order
        5. Show counts by each exp_group
        6. Show me average attack by exp_group where average attack is greater than 40

    - Furniture dataset:
        1. example mongodb queries
        2. example mongodb queries using aggregate
        3. example mongodb queries using sort
        4. Please give me the counts of the brands and sort by count in descending order
        5. Please give me average price by material for price less than 270

    - Used car dataset:
        1. Please give total age by model
        2. Please give average age by brand and sort by average age in descending order

## File Structure
- CSV Data (the csv data we used to test our sql portion)
    - CarPrice_Assignment.csv
    - Furniture.csv
    - uber.csv
- JSON Data (the json data we used to test our mongo portion)
    - furniture.json
    - pokedex.json
    - used_car.json
- chatdb.py (development file to test components of master_chatdb.py)
- master_chatdb.py (main file, the one used to run the main app)
- mongo_queries.py (file with mongo helper functions to get general and specific sample queries, as well as natural language queries)
- README.md
- requirements.txt (contains the requirements to run this project)
- sql_queries.py (file with sql helper functions for sample queries)