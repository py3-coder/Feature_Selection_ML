#!/usr/bin/env python
# coding: utf-8

# # Feature_Selection

# - **Having irrelevant features in your data can decrease the accuracy of the models and makes your models learn based on irrelevant**

# ## Defination

# **Feature Selection** :
# 
# - Process of selecting the best features which contribute maximum for the model in order to get best     result in term of accuracy or it should take less time for traning .
#         
# - Feature selection methods are intended to reduce the number of input variables to those that are believed to be most useful to a model in order to predict the target variable. 
# 
# 
# **Benefits of Performing Feature-Selection :**
# 
#     1.  Reduce Overfitting 
#     2.  Improve Accuracy
#     3.  Reduce Traning Time 
# 

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[4]:


pwd


# In[5]:


path='E:\\DataScience\\MachineLearning\\Breast-cancer-detection-using-ML'


# In[6]:


import os 
os.listdir(path)


# In[7]:


#reading data
df =pd.read_csv(path+"\\data.csv")


# In[8]:


df


# In[9]:


df.columns


# In[10]:


df.shape


# In[11]:


df.isnull().sum()


# In[12]:


df=df.drop(['Unnamed: 32','id'],axis=1)


# In[13]:


df


# # Univariate feature selection 

# <div class="alert alert-block alert-info">
#     <b>For regression</b>: f_regression, mutual_info_regression
#     <b>For classification</b>: chi2, f_classif, mutual_info_classif
# </div>

# In[14]:


#importing feauture_selection from sklearn
# stastistical fuction --- chi 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2 ,f_classif

#traning data -- all features except the target values
#here we are using  data shape -- 569 , 32
X =df.iloc[:,1:]
#target values
y =df.iloc[:,0]

# k= 15 : selecting top 15 values  which are highly co-related to taget 
# using chi2 function
fs_chi2 =SelectKBest(chi2 ,k=15)
X_chi2 = fs_chi2.fit_transform(X,y)

# k= 15 : selecting top 15 values  which are highly co-related to taget 
# using f_classif function
fs_f =SelectKBest(f_classif ,k=15)
X_f_classif = fs_f.fit_transform(X,y)


# In[15]:


#X_chi2 selected
dfscores = pd.DataFrame(fs_chi2.scores_)
dfcolumns =pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
#naming the dataframe columns
featureScores.columns = ['Specs','Score']
print(featureScores.nlargest(15,'Score')) 


# In[16]:


#X_f_classif Selected
dfscores = pd.DataFrame(fs_f.scores_)
dfcolumns =pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
#naming the dataframe columns
featureScores.columns = ['Specs','Score']
print(featureScores.nlargest(15,'Score')) 


# # Feature Importance

# - Feature importance gives you a score for each feature of your data, 
# - The higher the score more important or relevant is the feature towards your output variable.
# - Feature importance is an inbuilt class that comes with Tree Based Classifiers, 
# - We will be using Extra Tree Classifier for extracting the top 10 features for the dataset.

# In[17]:


from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()


# In[18]:


model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers


# In[19]:


#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()


# # Correlation Matrix with Heatmap

# - Correlation states how the features are related to each other or the target variable.
# - Correlation can be positive (increase in one value of feature increases the value of the target variable) or negative (increase in one value of feature decreases the value of the target variable)
# - Heatmap makes it easy to identify which features are most related to the target variable, we will plot heatmap of correlated features using the seaborn library.

# In[22]:


#get correlations of each features in dataset
corr = df.corr()
top_corr_features = corr.index
plt.figure(figsize=(20,20))
#plot heat map
plt=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[ ]:




