#!/usr/bin/env python
# coding: utf-8

# # Import
# 

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from IPython.display import Image,display
import matplotlib.pyplot as plt,pydotplus
from sklearn.metrics import accuracy_score
from sklearn import metrics


# # Code

# In[2]:


df = pd.read_csv(r"C:\Users\UTKARSKSH ANAND\OneDrive\Desktop\class.csv")
df.head()


# In[3]:


X = df.drop('Class attribute',axis='columns')
y = df['Class attribute']


# In[4]:


X


# In[5]:


y


# In[6]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


# In[7]:


X_train.shape, y_train.shape


# In[8]:


X_test.shape,y_test.shape


# In[9]:


clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)


# In[10]:


metrics.accuracy_score(y_test,y_pred)


# In[11]:


clf.score(X_train,y_train)


# In[12]:


clf.score(X_test,y_test)


# # Confusion matrix

# In[13]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)


# # Decision Tree

# In[14]:


train_accuracy = []
validation_accuracy = []
for depth in range(1,10):
    clf = DecisionTreeClassifier(max_depth=depth,max_leaf_nodes=10,random_state=10)
    clf.fit(X_train,y_train)
    train_accuracy.append(clf.score(X_train,y_train))
    validation_accuracy.append(clf.score(X_test,y_test))


# In[15]:


ddata=tree.export_graphviz(clf,out_file=None,filled=True,rounded=True,
                           feature_names=['Native English Speaker', 'Course Instructor',
                                         'Course','Summer or regular','Class size'],
                          class_names=['1','2','3'])
graph=pydotplus.graph_from_dot_data(ddata)
display(Image(graph.create_png()))


# In[ ]:




