############################################################################
# Association Rule Based Product Recommendations for Online Retail Customers
############################################################################

##############################################################
# 1. Business Problem
##############################################################
# Below are the basket information of 3 different users. Make product recommendations
# for these baskets using the association rule. The product recommendations can be one
# or more. Derive the decision rules based on the 2010-2011 Germany customers.
#       User 1's basket contains product ID: 21987
#       User 2's basket contains product ID: 23235
#       User 3's basket contains product ID: 22747

# Dataset Story
# The dataset named 'Online Retail II' includes online sales transactions of a UK-based
# retail company between 01/12/2009 and 09/12/2011. The product catalog of the company includes
# gift items, and it's known that most of its customers are wholesalers.

# Variables:
# InvoiceNo: Invoice number (If this code starts with 'C', it indicates that the transaction was canceled)
# StockCode: Product code (Unique for each product)
# Description: Product name
# Quantity: Quantity of the product sold in each transaction
# InvoiceDate: Invoice date
# UnitPrice: Unit price of the product in the invoice (in GBP - British Pounds)
# CustomerID: Unique customer number
# Country: Country name


###############################################################
# 2. Data Preparation
###############################################################

# Importing libraries
##############################################
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules
pd.set_option('display.max_colwidth', None)

df_ = pd.read_excel("online_retail_II.xlsx",sheet_name="Year 2010-2011", engine='openpyxl')
df = df_.copy()


# Data understanding
##############################################
def check_df(dataframe, head=5):
    print('################# Shape ################# ')
    print(dataframe.columns)
    print('################# Types  ################# ')
    print(dataframe.dtypes)
    print('##################  Head ################# ')
    print(dataframe.head(head))
    print('#################  Shape ################# ')
    print(dataframe.shape)
    print('#################  NA ################# ')
    print(dataframe.isnull().sum())
    print('#################  Quantiles ################# ')
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99]).T)

check_df(df)


# Calculating the lower and upper thresholds for identifying outliers in a given variable
#########################################################################################
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


# Surpressing outlier values of a given variable of the dataframe
##################################################################
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Data cleaning
##############################################################
def retail_data_prep(dataframe):
    # Dropping rows with missing values
    dataframe.dropna(inplace=True)
    # Removing the rows from the dataframe where the 'StockCode' column has the value 'POST
    dataframe = dataframe[dataframe['StockCode'] != 'POST']
    # Removing transactions with 'C' in the Invoice column (indicating canceled transactions)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    # Selecting rows with 'Quantity' and 'Price' greater than 0
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    # Examining outliers and if necessary, performing outlier suppression
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

df = retail_data_prep(df)


##########################################
# 3. Association Rules
##########################################

# # Creating a pivot table for 'Invoice' and 'Product'
# #######################################################
def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


# Creating association rules
##############################
def create_rules(dataframe, id=True, country="Germany"):
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_df(dataframe, id)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules

rules = create_rules(df)


# Creating the 'arl_recommender' function and make product recommendations for given users
##########################################################################################
#       User 1's basket contains product ID: 21987
#       User 2's basket contains product ID: 23235
#       User 3's basket contains product ID: 22747

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]

# User 1
arl_recommender(rules, 21987, 1) # --> 21086
# User 2
arl_recommender(rules, 23235, 1) # --> 23244
# User 3
arl_recommender(rules, 22747, 1) # --> 22745


# Retrieve the product name from the product ID
###############################################
def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

# User 1
check_id(df, 21086) # --> ['SET/6 RED SPOTTY PAPER CUPS']
# User 2
check_id(df, 23244) # --> ['ROUND STORAGE TIN VINTAGE LEAF']
# User 3
check_id(df, 22745) # --> ["POPPY'S PLAYHOUSE BEDROOM "]