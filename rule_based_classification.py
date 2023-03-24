import pandas as pd
def import_csv(dataframe):
    df = pd.read_csv(dataframe)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 500)
    pd.set_option('display.html.table_schema', True)
    return df
df = import_csv("Sales.csv")

# General image
def check_df(dataframe):
    print("#### Shape ####")
    print(dataframe.shape)
    print("#### Columns ####")
    print(dataframe.columns)
    print("#### Types ####")
    print(dataframe.dtypes)
    print("#### NA ####")
    print(dataframe.isnull().sum())
    print("#### Quantiles ####")
    print(dataframe.describe([0, 0.05, 0.95, 0.99, 1]).T)
check_df(df)

# Find outliers on the variables.
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.25)
    quartile3 = dataframe[variable].quantile(0.75)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# Suppress low limit, upper limit with outliers.
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe = dataframe[dataframe["Profit"] > 0]
    replace_with_thresholds(dataframe, "Unit_Cost")
    replace_with_thresholds(dataframe, "Unit_Price")
    replace_with_thresholds(dataframe, "Profit")
    replace_with_thresholds(dataframe, "Cost")
    replace_with_thresholds(dataframe, "Revenue")
    return dataframe

df = retail_data_prep(df)
df.describe().T
df.head()

#############################################
# RULE BASED CLASSIFICATION FOR POTENTIAL CUSTOMER REVENUE
#############################################

#############################################
# Business Problem
#############################################

# A bike company wants to use some characteristics of its customers to develop new customer definitions
# based on level, develop segments in accordance with these new customer definitions,
# and calculate the average earnings of the new customers in accordance with these segments.


df["Customer_Gender"].value_counts()
df["Country"].value_counts()
df["Product_Category"].value_counts()
df["Revenue"].value_counts()


df.groupby("Customer_Gender").agg({"Revenue": ["mean", "count", "sum"]})
df.groupby("Country").agg({"Revenue": ["mean", "count", "sum"]})
df.groupby(["Country", "Customer_Gender"]).agg({"Revenue": "mean"})
df.groupby(["Country", "Product_Category"]).agg({"Revenue": "mean"})

#############################################
# What are the average revenue in the breakdown of Country, Product_Category, Customer_Gender, Customer_Age?
# If the average is sorted according to the revenue in descending order, it can be seen how much it brings on average.
agg_df = df.groupby(["Country", "Product_Category", "Customer_Gender", "Customer_Age"]).agg({"Revenue": "mean"}).sort_values("Revenue", ascending=False)
agg_df.reset_index(inplace=True)

#############################################
# Here, the groups to which the ages will be separated can be divided into a certain sector information.
# For this study, division by quartiles was preferred.
# The variable added by creating a range was converted into a categorical variable.

agg_df["Age_Cat"] = pd.cut(agg_df["Customer_Age"], bins=[0, 17, 28, 35, 43, 87], labels=["0-17", "18-27", "28-34", "35-42", "43-87"])
agg_df["Age_Cat"] = agg_df["Age_Cat"].astype(str)

#############################################
# Create and add to dataset a new variable named customers_level_based.
# After that for singularization, do groupby according to customers_level_based, take the revenue mean.

agg_df["customers_level_based"] = agg_df[["Country", "Product_Category", "Customer_Gender", "Age_Cat"]].apply(lambda x: "_".join(x).upper(), axis=1)
agg_df = agg_df.groupby("customers_level_based").agg({"Revenue": "mean"})

#############################################
# Segmentation is done according to Revenue.

agg_df["SEGMENT"] = pd.qcut(agg_df["Revenue"], 4, labels=["D", "C", "B", "A"])
agg_df.groupby("SEGMENT").agg({"Revenue": ["mean", "sum", "count"]})
agg_df.reset_index(inplace=True)

#############################################
# Find out how much a new customer can bring in on average.

# What segment does a 45-year-old Australian woman who buys Bicycle Accessories belong to and
# how much income on average is she expected to earn?

new_user = "AUSTRALIA_ACCESSORIES_F_43-87"
agg_df[agg_df["customers_level_based"] == new_user]
