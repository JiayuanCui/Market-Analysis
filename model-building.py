import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import shap

new_df = pd.read_csv('new-dataframe.csv')

# use random forest
# drop ID as everyone has unique ID
rd_df = new_df.drop(columns=['ID', 'Dt_Customer'])
rd_df.replace([np.inf, -np.inf], 0, inplace=True)

# One-hot encoding
rd_df = pd.get_dummies(rd_df)

X=rd_df.drop(columns=['NumStorePurchases'])  # Features
y=rd_df['NumStorePurchases']  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) 
# 70% training and 30% test

#Create a Random Forest Classifier with 100 trees
rg = RandomForestRegressor(n_estimators=200, n_jobs=-1)

#Train the model using the training sets y_pred=clf.predict(X_test)
rg.fit(X_train, y_train)

y_pred=rg.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# The range of NumStorePurchases is 13, and the Root Mean Squared Error is only 1.1(less than 10% of the range), 
# which means it is a relaible model.

# find feature importance scores
feature_imp = pd.Series(rg.feature_importances_, 
        index = list(X.columns)).sort_values(ascending=False)

feature_imp = feature_imp[:10]

# Creating a bar plot
plt.figure(figsize = (7, 7))
sns.barplot(x=feature_imp[:10], y=feature_imp.index[:10])
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Top 10 Important Features")
plt.savefig('important_feautres.png', bbox_inches='tight')
plt.show()

# calculate shap values 
ex = shap.Explainer(rg, X_train)
shap_values = ex(X_test)

# plot
plt.title('SHAP summary for NumStorePurchases', size=16)
fig = shap.plots.beeswarm(shap_values, max_display=8)
plt.savefig('SHAP.png', bbox_inches='tight')
plt.show()

# Saving the model
import pickle
pickle.dump(rg, open('rg.pkl', 'wb'))


#Creating a bar plot using Streamlit
# fig, ax = plt.subplots(figsize=(7,7))
# sns.barplot(x=load_rg.feature_imp[:10], y=load_rg.feature_imp.index[:10])
# ax.set_xlabel('Feature Importance Score')
# ax.set_ylabel('Features')
# ax.set_title("Visualizing Top 10 Important Features")
# st.pyplot(fig)

# calculate shap values 
# ex = shap.Explainer(rg, X_train)
# shap_values = ex(X_test)

# plot
# plt.title('SHAP summary for NumStorePurchases', size=16)
# st_shap(shap.plots.beeswarm(shap_values, max_display=8))