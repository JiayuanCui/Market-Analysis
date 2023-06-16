from turtle import position
import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
import io 
from scipy.stats import pearsonr
import pickle
import shap
from streamlit_shap import st_shap
from scipy.stats import ttest_ind 

st.set_page_config()

st.title('Customer Personality Analysis')
st.markdown("""
Customer Personality Analysis is a detailed analysis of a company’s ideal customers. 
It helps a business to better understand its customers and makes it easier for them 
to modify products according to the specific needs, behaviors and concerns of different 
types of customers.

Customer personality analysis helps a business to modify its product based on its 
target customers from different types of customer segments. For example, instead of 
spending money to market a new product to every customer in the company’s database, 
a company can analyze which customer segment is most likely to buy the product and then 
market the product only on that particular segment.

**Data source:** https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis.
""")

#load data
df = pd.read_csv('marketing_campaign.csv')

st.header("Marketing Campaign Data")
st.write(df)

expander_bar = st.expander("Attributes")
expander_bar.markdown("""
**People**

ID: Customer's unique identifier

Year_Birth: Customer's birth year

Education: Customer's education level

Marital_Status: Customer's marital status

Income: Customer's yearly household income

Kidhome: Number of children in customer's household

Teenhome: Number of teenagers in customer's household

Dt_Customer: Date of customer's enrollment with the company

Recency: Number of days since customer's last purchase

Complain: 1 if the customer complained in the last 2 years, 0 otherwise

**Products**

MntWines: Amount spent on wine in last 2 years

MntFruits: Amount spent on fruits in last 2 years

MntMeatProducts: Amount spent on meat in last 2 years

MntFishProducts: Amount spent on fish in last 2 years

MntSweetProducts: Amount spent on sweets in last 2 years

MntGoldProds: Amount spent on gold in last 2 years

**Promotion**

NumDealsPurchases: Number of purchases made with a discount

AcceptedCmp1: 1 if customer accepted the offer in the 1st campaign, 0 otherwise

AcceptedCmp2: 1 if customer accepted the offer in the 2nd campaign, 0 otherwise

AcceptedCmp3: 1 if customer accepted the offer in the 3rd campaign, 0 otherwise

AcceptedCmp4: 1 if customer accepted the offer in the 4th campaign, 0 otherwise

AcceptedCmp5: 1 if customer accepted the offer in the 5th campaign, 0 otherwise

Response: 1 if customer accepted the offer in the last campaign, 0 otherwise

**Place**

NumWebPurchases: Number of purchases made through the company’s website

NumCatalogPurchases: Number of purchases made using a catalogue

NumStorePurchases: Number of purchases made directly in stores

NumWebVisitsMonth: Number of visits to company’s website in the last month

**Target**

Need to perform clustering to summarize customer segments.
""")

# understanding data
st.subheader("Understanding Data")
def basic_info(df):
    st.write("This dataset has ", df.shape[1], " columns and ", df.shape[0], " rows.")
    st.write("This dataset has ", df[df.duplicated()].shape[0], " duplicated rows.")
    st.write(" ")
    expander_bar1 = st.expander("Descriptive statistics")
    expander_bar1.write(df.describe())
    expander_bar2 = st.expander("Information about this dataset")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    expander_bar2.text(s)

st.write(basic_info(df))

# data cleaning
st.subheader("Data Cleaning")
df_copy = df.copy()

# divide the data into two dataframes: one has income values, and the other doesn't.
have_income = df_copy[df_copy.Income.isnull()==False]
missing_income = df_copy[df_copy.Income.isnull()==True]

# impute the missing values with the median of Income
missing_income.Income = have_income.Income.median()
df_copy = missing_income.append(have_income)

# convert string type into datetime type
df_copy.Dt_Customer = pd.to_datetime(df_copy.Dt_Customer)

st.markdown("""
**Observation**

* The Income column has 24 missing values
Solution: impute the missing values with the median of Income. 

* Dt_Customer's type is string 
Solution: convert string type into datetime type.
""")

# test
buffer = io.StringIO()
df_copy.info(buf=buffer)
s = buffer.getvalue()
expander_bar3 = st.expander("New information about this dataset")
expander_bar3.text(s)

st.markdown("""**New Dataframe**""")
st.write(df_copy)

# store and download the file
def filedownload(df):
    df.reset_index(drop=True)
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="clean-data.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(df_copy), unsafe_allow_html=True)

# PART 1: EDA
st.header("Exploratory Data Analysis")
df = df_copy
# check outliers
st.subheader("Outliers")

# select columns to plot
df_to_plot = df.drop(columns=['ID', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 
'AcceptedCmp5', 'Response', 'Complain', 'Z_CostContact', 'Z_Revenue']).select_dtypes(include=np.number)
# subplots
fig, axs = plt.subplots(4,4,figsize=(12,14))
for i, ax in enumerate(axs.flat):
    ax.boxplot(df_to_plot.iloc[:,i], patch_artist=True)
    ax.set_title(df_to_plot.columns[i]) 
st.pyplot(fig)

st.markdown("""
I decided to remove the outliers in the Year_birth column since nobody who was born before 1900 can be still alive.
""")

col1, col2 = st.columns((1,1))
expander_bar4 = col1.expander("Old Descriptive statistics")
expander_bar4.write(df.Year_Birth.describe())
# Remove outliers in Year_birth
new_df = df[df.Year_Birth >= (df.Year_Birth.mean()-3*df.Year_Birth.std())]
expander_bar5 = col2.expander("""New Descriptive statistics""")
expander_bar5.write(new_df.Year_Birth.describe())

# Create new features
new_df["Join_year"] = new_df.Dt_Customer.dt.year
new_df["Join_month"] = new_df.Dt_Customer.dt.month
new_df["Join_weekday"] = new_df.Dt_Customer.dt.weekday
new_df["Minorhome"] = new_df.Kidhome + new_df.Teenhome
new_df['Total_Mnt'] = new_df.MntWines+ new_df.MntFruits+ new_df.MntMeatProducts+ new_df.MntFishProducts+ new_df.MntSweetProducts+ new_df.MntGoldProds 
new_df['Total_num_purchase'] = new_df.NumDealsPurchases+ new_df.NumWebPurchases+ new_df.NumCatalogPurchases+ new_df.NumStorePurchases+ new_df.NumWebVisitsMonth 
new_df['Total_accept'] = new_df.AcceptedCmp1+ new_df.AcceptedCmp2+ new_df.AcceptedCmp2+ new_df.AcceptedCmp2+ new_df.AcceptedCmp3+ new_df.AcceptedCmp4+ new_df.AcceptedCmp5+ new_df.Response
new_df['AOV'] = new_df.Total_Mnt/new_df.Total_num_purchase

st.subheader("Create a new dataframe")
expander_bar6 = st.expander("New Attributes")
expander_bar6.markdown("""
Join_year: customer's enrolled year

Join_month: customer's enrolled month

Join_weekday: customer's enrolled weekday

Minorhome: Number of minors in customer's household

Total_Mnt: Total amount spent in the last 2 years

Total_num_purchase: Total number of purchases made in the last 2 years

Total_accept: Total amount a customer accepted the offer in marketing campaign

AOV: Average order volumn of each customer (Total_Mnt/Total_num_purchase)
""")

columns = new_df.columns.tolist()
selected_cols = st.multiselect("Select the columns to display", columns)
if len(selected_cols) > 0:
    selected_df = new_df[selected_cols]
    st.write(selected_df)
else:
    st.write(new_df)

# store and download the file
def filedownload(df):
    df.reset_index(drop=True)
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="new-dataframe.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(new_df), unsafe_allow_html=True)

# patterns and anomalies
st.subheader("Patterns & Anomalies")
st.markdown("""
**Patterns:**

1. High-Income People
* tend to spend more and purchase more.
* tend to visit the company's website less frequently than other people.
* tend to have a few numbers of purchases made with a discount.

2. People having kids at home
* tend to spend less and purchase less.
* tend to have a high number of purchases made with a discount.

3. People who purchased with high average order volume
* tend to buy more wines and meat products.
* tend to make a high number of purchases made using a catalog.
* tend to not visit the company's website.
""")

# select columns to plot
df_to_plot = new_df.drop(columns=['ID', 'Z_CostContact', 'Z_Revenue'])

# create heatmap
fig, ax = plt.subplots(figsize = (12, 9))
sns.heatmap(df_to_plot.corr(), cmap = 'RdBu',vmin = -1, vmax = 1,center = 0)
ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 12)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, fontsize = 12)
plt.title("Correlation Heatmap", fontsize=18)
st.pyplot(fig)

# select columns to plot
plt.figure(figsize = (30, 20))
s = sns.clustermap(df_to_plot.corr(method = 'kendall'),  cmap = 'RdBu',vmin = -1, vmax = 1,center = 0)
plt.title("Correlation Heatmap", fontsize=18)
st.pyplot(s)

st.write('---')
st.markdown("""
**Anomalies:**

Usually, the complaints of a customer are negatively correlated to the total amount spent. 
However, according to the person correlation, the number of complaints has almost no correlation 
with the total amount spent in the last two years, since we only have 20 out of 2200 customers 
who complained in the last two years. Therefore, the customer service in the company has done 
a wonderful job in the last two years.
""")

# Visualize Complain vs Total_Mnt
fig, ax = plt.subplots(figsize=(10,6))
plt.scatter(x=new_df.Complain, y=new_df.Total_Mnt)
ax.set_xlabel("complain")
ax.set_ylabel("Total amount spent in the last 2 years")
plt.title("Complain vs Total_Mnt")
st.pyplot(fig)

st.write("There are", new_df[new_df.Complain > 0].ID.nunique(), "complaints in total.")

r, p_value  =  pearsonr(x=new_df['Complain'], y=new_df['Total_Mnt'])
# print results
st.write('Pearson correlation: ', r)

st.write('---')
st.markdown("""Intuitively, the number of visits to company’s website in the last month 
will affect the number of purchases made through the company’s website. However, according to 
the scatter plot, there is no correlation between NumWebPurchases and NumWebVisitsMonth.
""")
# Visualize NumWebPurchases vs NumWebVisitsMonth
fig, ax = plt.subplots(figsize=(10,6))
plt.scatter(x=new_df.NumWebPurchases, y=new_df.NumWebVisitsMonth)
ax.set_xlabel("Number of purchases made through the company’s website")
ax.set_ylabel("Number of visits to company’s website in the last month")
plt.title("NumWebPurchases vs NumWebVisitsMonth")
st.pyplot(fig)

# PART 2: Statistical Analysis
# predict the number of store purchases
st.header("Statistical Analysis")
st.subheader('Predict the Number of Store Purchases')
fig, ax = plt.subplots(figsize=(10,6))
ax.hist(new_df.NumStorePurchases)
ax.set_xlabel("Number of purchases made directly in stores")
ax.set_ylabel("Count")
ax.set_title("Distribution of the number of store purchases")
st.pyplot(fig)

# 
# Reads in saved classification model
load_rg = pickle.load(open('rg.pkl', 'rb'))

st.image('important_feautres.png')
st.image('SHAP.png')

st.markdown("""
For predicting the number of purchases made directly in stores, top 7 factors are: 

* Total_Mnt: Total amount spent in the last 2 years
* AOV: Average order volumn of each customer
* Total_num_purchase: Total number of purchases made in the last 2 years
* NumCatalogPurchases: Number of purchases made using a catalogue
* NumWebPurchases: Number of purchases made through the company’s website
* NumWebVisitsMonth: Number of visits to company’s website in the last month
* MntWines: Amount spent on wine in last 2 years

1. The number of store purchases increases with the higher total amount spent, higher total number of purchases, higher AOV, and higher amount spent on wine. 
2. The number of store purchases decreases with the higher number of visits to company’s website, the higher number of purchases made using a catalogue, and the higher number of purchases made through the company’s website.

**Summary:** People who mostly shop at stores tend to buy more wines, have a higher average order volume, 
and shop less through the websites or catalogues.
""")

st.write("---")
st.markdown("**In-store shoppers VS. Average order volume**")
store_shoppers = new_df[new_df.NumStorePurchases>0]
store_shoppers = store_shoppers[store_shoppers.AOV <= (store_shoppers.AOV.mean()+3*store_shoppers.AOV.std())]
store_shoppers['Type of shopper'] = "In-store"
other_shoppers = new_df[new_df.NumStorePurchases==0]
other_shoppers['Type of shopper'] = "Other"

col3, col4 = st.columns((1,1))
fig, ax = plt.subplots(figsize=(5,6))
all_shoppers = store_shoppers.append(other_shoppers)
sns.boxplot(data = all_shoppers, x = 'Type of shopper', y = 'AOV')
ax.set_ylabel("Average Order Volume")
ax.set_title("Type of Shopper vs NumStorePurchases")
col3.pyplot(fig)

# Visualize AOV vs NumStorePurchases
fig, ax = plt.subplots(figsize=(5,6))
plt.scatter(x=all_shoppers.AOV, y=all_shoppers.NumStorePurchases)
ax.set_xlabel("AOV")
ax.set_ylabel("NumStorePurchases")
ax.set_title("AOV vs NumStorePurchases")
col4.pyplot(fig)

all_shoppers.replace([np.inf, -np.inf], 0, inplace=True)
r, p_value  =  pearsonr(x=all_shoppers['AOV'], y=all_shoppers['NumStorePurchases'])

# print results
st.write('Pearson correlation: ', r)
st.write('Pearson p-value: ', p_value)

st.write("""According to the scatter plot, a higher number of in-store purchases tends to have a higher average order volume. 
Also, the Pearson correlation of""", r, """and a p-value of""", p_value, """indicate that they are statistically significant 
and have a positive correlation.
""")

st.markdown("**Gold Purchases Amount VS. Number of In-Store Purchases**")
gold_above_avg =  new_df[new_df.MntGoldProds > new_df.MntGoldProds.mean()]
gold_above_avg['Gold Purchases Amount'] = "Above Average"
gold_equ_or_below_avg =  new_df[new_df.MntGoldProds <= new_df.MntGoldProds.mean()]
gold_equ_or_below_avg['Gold Purchases Amount'] = "Equals or Below Average"

col5, col6 = st.columns((1,1))
fig, ax = plt.subplots(figsize=(5,6))
df_gold = gold_above_avg.append(gold_equ_or_below_avg)
sns.boxplot(data = df_gold, x = 'Gold Purchases Amount', y = 'NumStorePurchases')
ax.set_ylabel("Number of In-Store Purchases")
ax.set_title("2 Gold Purchases Groups")
col5.pyplot(fig)

# Visualize MntGoldProds vs NumStorePurchases
fig, ax = plt.subplots(figsize=(5,6))
plt.scatter(x=new_df.MntGoldProds, y=new_df.NumStorePurchases)
ax.set_xlabel("MntGoldProds")
ax.set_ylabel("NumStorePurchases")
ax.set_title("MntGoldProds vs NumStorePurchases")
col6.pyplot(fig)

r, p_value  =  pearsonr(x=new_df['MntGoldProds'], y=new_df['NumStorePurchases'])

# print results
st.write('Pearson correlation: ', r)
st.write('Pearson p-value: ', p_value)

st.write("""
There is a trend that as MntGoldProds increases, NumStorePurchases also increase. Also, the Pearson correlation of""", 
r, """and a p-value of""", p_value, """indicates that they are statistically significant and have a positive correlation.
We can conclude that people who buy gold are more conservative. Therefore, people who spent an above-average amount on gold 
in the last 2 years would have more in-store purchases.""")

# PART 3: Data Visualization
st.header("Data Visualization")
st.markdown("**Marketing Campaigns & Marketing Channels**")
col7, col8 = st.columns((1,1))

fig, ax = plt.subplots(figsize=(5,6))
new_df[["AcceptedCmp1", "AcceptedCmp2","AcceptedCmp3","AcceptedCmp4","AcceptedCmp5","Response"]].sum().sort_values().plot.barh()
plt.xlabel("Number of Offer Accepted")
plt.ylabel("Campaign")
col7.pyplot(fig)

fig, ax = plt.subplots(figsize=(5,6))
new_df[["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]].sum().sort_values().plot.barh()
plt.xlabel("Total number of purchases")
plt.ylabel("Channel")
col8.pyplot(fig)

st.markdown("The last marketing campaign is most successful, and Catalogue is the most underperforming channel.")

st.subheader("Average Customer")
new_df.replace([np.inf, -np.inf], 0, inplace=True)
st.write(new_df.mean())
st.markdown("""
An average customer
* has an annual income of 52227 dollars
* has an AOV of 26.8 dollars
* has purchased 20 times in the past 2 years
* became a customer in mid-June
* became a customer on Thursday
""")

st.subheader("Last Campaign")
# create 2 groups that accepted the offers from the last campaign and the campaign 1-5
cp_last = new_df[new_df.Response > 0]
cp__the_rest = new_df[new_df.AcceptedCmp2 == 0]

# remove the overlapping customers who accepted offers from both cp_last and cp__the_rest 
# so that twe can see the clear differences between these two groups
cp__the_rest2 = cp__the_rest
for i in list(cp__the_rest.ID):
    if i in list(cp_last.ID):
        cp__the_rest2 = cp__the_rest2[cp__the_rest2.ID != i]

cp_last = cp_last[['Year_Birth', 'Income', 'Minorhome', 'Join_month', 'Join_weekday',
                  'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
                  'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
                  'Total_Mnt', 'Total_num_purchase', 'AOV']]
cp__the_rest2 = cp__the_rest2[['Year_Birth', 'Income', 'Minorhome', 'Join_month', 'Join_weekday',
                  'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
                  'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
                  'Total_Mnt', 'Total_num_purchase', 'AOV']]

new_df2 = new_df[['Year_Birth', 'Income', 'Minorhome', 'Join_month', 'Join_weekday',
                  'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
                  'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
                  'Total_Mnt', 'Total_num_purchase', 'AOV']]

# visualize the differences
fig, ax = plt.subplots(figsize=(9,7))
value1 = pd.DataFrame((((cp_last.mean()) - new_df2.mean()) / new_df2.mean())*100)
value1.dropna(inplace = True)
value1.sort_values(by=0,inplace = True)
value1['positive'] = value1[0] >=0
value1[0].plot(kind='barh', color=value1.positive.map({True: 'navy', False: 'orange'}))
ax.set_title("Customer Characteristics Comparison - Customer in last campaign vs Average customer")
ax.set_xlabel("Difference in %")
ax.set_ylabel("Characteristics")
st.pyplot(fig)

# visualize the differences
fig, ax = plt.subplots(figsize=(9,7))
value = pd.DataFrame((((cp_last.mean()) - cp__the_rest2.mean()) / cp__the_rest2.mean())*100)
value.dropna(inplace = True)
value.sort_values(by=0,inplace = True)
value['positive'] = value[0] >=0
value[0].plot(kind='barh', color=value.positive.map({True: 'navy', False: 'orange'}))
ax.set_title("Customer Characteristics Comparison - The last campaign vs Campaign 1-5")
ax.set_xlabel("Difference in %")
ax.set_ylabel("Characteristics")
st.pyplot(fig)

st.write("---")
# Reads in saved classification model
load_rg2 = pickle.load(open('rg2.pkl', 'rb'))

st.image('important_feautres2.png')
    
st.markdown("""
The last campaign attracted more valuable customers in terms of AOV, the total amount spent, 
and the total number of purchases compared to the customers attracted by the previous campaigns.

The customers in the last campaign spent nearly two times more money on meat products and wines 
compared to the customers in the previous campaigns.

The customers in the last campaign purchased more evenly through stores, websites, and catalogs, 
whereas the customers in the previous campaigns mostly purchased through stores and websites.

The customers in the last campaign earned 20% more salary than the customers in the previous campaigns.
""")