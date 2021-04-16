#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
            print(os.path.join(dirname, filename))
# Any results you write to the current directory are saved as output.


# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns
dataset=pd.read_csv("../input/train.csv")
X_test= pd.read_csv("../input/test.csv")
dataset.head()


# * The dataset here is a sample of the transactions made in a retail store. 
# * The store wants to know better the customer purchase behaviour against different products. 
# * Specifically, here the problem is a regression problem where we are trying to predict the dependent variable (the amount of purchase) with the help of the information contained in the other variables.
# * There are seven categorical variables to analyse.
# 
# 

# Let us list down some points that can be addressed with the analsysis.
# 1. Understanding the cutomers on the basis of their purchasing habits.
# 2. Understanding the purchasing habits according to Age groups, Occuptation, City_Categories.
# 3. The above segmented group of users can be then used to model the data and use to predict the purchase spend for each customer. 
# Lets dive in by understanding the data.
# 
# 

# In[4]:


dataset.info()


# **** There are null values in Product_category_2, Product_Category_3

# In[5]:


dataset.describe()


# Mean value of Product_Category_2 is 9.8 and that for Product_Category_3 is 12.6, which we will
# use to fill the missing values in these two columns.

# In[6]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(dataset.iloc[:, 9:11].values)
dataset.iloc[:,9:11] = imputer.transform(dataset.iloc[:, 9:11].values)
dataset.info() 


# In[7]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(X_test.iloc[:, 9:11].values)
X_test.iloc[:,9:11] = imputer.transform(X_test.iloc[:, 9:11].values)
X_test.info() 


# * Dropping the columns that intuitiey should not imapact the purchase outcome, i.e. User_ID and Product ID. 

# In[8]:


dataset.drop(['User_ID','Product_ID'], axis=1, inplace=True)
dataset.info()


# In[9]:


X_test.drop(['User_ID','Product_ID'], axis=1, inplace=True)
X_test.info()


# In[10]:


dataset.head()


# > There are still some special characters, like (+) in the columns 'Age' and 'stay in Current City_Years, which need to be removed, before machine learning algorithms can be run later.

# In[11]:


dataset['Age']=(dataset['Age'].str.strip('+'))
X_test['Age']=(X_test['Age'].str.strip('+'))


# In[12]:


dataset['Stay_In_Current_City_Years']=(dataset['Stay_In_Current_City_Years'].str.strip('+').astype('float'))
X_test['Stay_In_Current_City_Years']=(X_test['Stay_In_Current_City_Years'].str.strip('+').astype('float'))


# * Let us inspect the data now.

# In[13]:


dataset.info()
dataset.head()


# * As can be seen, we have managed to clean the columns to our reuirement and removed the '+' sign from the two columns. 
# > At this stage, I will exploratory data analysis by visualising the data, in particular, by visualising the statistical relationship between the different variables.

# > Exploratory data analysis supported by data visualisations.

# In[14]:


sns.heatmap(
    dataset.corr(),
    annot=True
)


# > The key take aways from the above plot are the positive correlation coefficients of three features as a function of Purchase:
# * Occupation
# * Stay_In_Current_City_Years
# * Marital Status
# 
# > Increase in any of the values for the above three features is likey to result in a higher purchase from the customer.

# In[15]:


g = sns.FacetGrid(dataset,col="Stay_In_Current_City_Years")
g.map(sns.barplot, "Marital_Status", "Purchase");


# > It is difficult to conclude anything from the above visulaisation, but it might be useful to analyse if the trend shows something different across the different cities. 

# In[16]:


sns.jointplot(x='Occupation',y='Purchase',
              data=dataset, kind='hex'
             )


# 1.  First insight would be that most of the purchase is done between 5000-10000. 
# 2. Next important insight, would be the occupations that lead to highest purchases. In this case, it would be occupation 4, listed in the dataset, closely followed that by 0 and 7.
# > One can imagine that the store can run targeted advertiements next time around to people with above listed occupations as they more likely to spend within the above purchase range.
# 
# **To get a better understanding, we will now analyse the purchase habits across the different city categories.**

# In[17]:


g = sns.FacetGrid(dataset,col="City_Category")
g.map(sns.barplot, "Gender", "Purchase");


# 1. Clearly people from City_Category C are showing higher purchase capacity as compared to the other two cities on average.
# 2. For City_categories B and C, Males tend to dominate the purchasing, whereas it is the opposite for City Category_C, where Females tend to puchase more than men. It is a useful insight, and it be useful to oserve which age group of females does higher purchasing.

# In[18]:


g = sns.FacetGrid(dataset,col="Age",row="City_Category")
g.map(sns.barplot, "Gender", "Purchase");


# * So, we focus on the first row of the visuaisation, i.e. City_Category_A and then on the bar for females. 
# > There are two age groups that can be identified with higher purchase, 26-35 and 18-25. 
# Therefore, apart from the male population of all the three city categories, females of City Category A in the above two identifies age groups can be identified as potential buyers for next time around.

# In[19]:


sns.violinplot(x='City_Category',y='Purchase',hue='Marital_Status',
               data=dataset)


# * Ananlysis of Purchase capacity as a function of Marital Status across city categories does not show a definitive trend. It would lead to a lot of assumptions and might lead to wrong conlcusions.

# In[20]:


fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(18,14))
ax = sns.pointplot(y='Product_Category_1', x='City_Category',hue='Age',
                 data=dataset,
                 ax=axes[0,0]
                )
ax = sns.pointplot(y='Product_Category_2', x='City_Category',hue='Age',
                 data=dataset,
                 ax=axes[0,1]
                )
ax = sns.pointplot(y='Product_Category_3', x='City_Category', hue='Age',
                 data=dataset,
                 ax=axes[1,0]
                )
ax = sns.pointplot(y='Purchase', x='City_Category', hue='Age',
                 data=dataset,
                 ax=axes[1,1]
                )


# Picking one key highlight from the above visualisation:
# > The stark difference in the purchase acoss City_Categories for the Age Group of 55 and above. It is highest in City_Category_B, as compared to the other age groups which tend to show high purchase in City_Category_C. 

# Having listed down the insights from each step above, let us now move to the next stage of the project, i.e data modelling and predication of sales.

# In[21]:


#Dividing the data into test and train datasets
X_train = dataset.iloc[:, 0:9].values
y_train = dataset.iloc[:, 9].values


# Let us inspet each of the split datasets

# In[22]:


X_train


# In[23]:


y_train


# In[24]:


X_test=X_test.values


# For X_rrain and X_test, there are categorical variables, which need to be encoded before they can be incorporated into the data model. We will convert each of the variable step by step and cross check our results.

# In[25]:


from sklearn.preprocessing import LabelEncoder
labelencoder_X_train = LabelEncoder()
X_train


# In[26]:


X_train[:, 0] = labelencoder_X_train.fit_transform(X_train[:, 0])
X_train


# In[27]:


X_train[:, 1] = labelencoder_X_train.fit_transform(X_train[:, 1])
X_train


# In[28]:


X_train[:, 3] = labelencoder_X_train.fit_transform(X_train[:, 3])
X_train


# > Doing the same steps for the X_test dataset

# In[29]:


labelencoder_X_test = LabelEncoder()
X_test


# In[30]:


X_test[:, 0] = labelencoder_X_test.fit_transform(X_test[:, 0])
X_test


# In[31]:


X_test[:, 1] = labelencoder_X_test.fit_transform(X_test[:, 1])
X_test


# In[32]:


X_test[:, 3] = labelencoder_X_test.fit_transform(X_test[:, 3])
X_test


# Having encoded the features, in the next step we will scale all the features to avoid issues due to different measurement scales.

# In[33]:


# Feature Scaling of training and test set
from sklearn.preprocessing import StandardScaler
sc_X_train = StandardScaler()
X_train = sc_X_train.fit_transform(X_train)

sc_X_test = StandardScaler()
X_test = sc_X_test.fit_transform(X_test)


# In[34]:


#Fitting the model
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error


# Fitting the model with the best number of n_estimators, to avoid underfitting and overfitting.

# In[35]:


regressor = RandomForestRegressor(n_estimators=700, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


# Lets us see what our prdicted results look like.

# In[36]:


y_pred


# In[ ]:


test=pd.read_csv("../input/test.csv")


# In[ ]:


submission = pd.DataFrame({
        "Purchase": y_pred,
        "User_ID": test["User_ID"],
        "Product_ID": test["Product_ID"]
        
    })
submission.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:




