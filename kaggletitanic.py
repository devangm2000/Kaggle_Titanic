#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


train=pd.read_csv("train.csv")
test= pd.read_csv('test.csv')


# In[3]:


gender_submission=pd.read_csv('gender_submission.csv') 


# In[4]:


#replacing null values
train['Age'].fillna(train['Age'].mean(),inplace=True)
test['Age'].fillna(test['Age'].mean(),inplace=True)
test['Fare'].fillna(test['Fare'].mean(),inplace=True)
train['Embarked'].fillna(value='S',inplace=True)


# In[30]:


train


# In[6]:


test


# In[7]:


train['family']=train['SibSp']+train['Parch']+1
test['family']=test['SibSp']+train['Parch']+1


# In[8]:


#encoding categorical data
train['Sex'] = train['Sex'].replace(['female','male'],[0,1])
train['Embarked'] = train['Embarked'].replace(['S','Q','C'],[1,2,3])

test['Sex'] = test['Sex'].replace(['female','male'],[0,1])
test['Embarked'] = test['Embarked'].replace(['S','Q','C'],[1,2,3])


# In[36]:


train


# In[37]:


test


# In[32]:


X_train=train.drop(columns=['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Survived'])
X_train


# In[38]:


y_train=train[['Survived']]
y_train


# In[39]:


X_test=test.drop(columns=['PassengerId','Name','SibSp','Parch','Ticket','Cabin'])
X_test


# In[40]:


y_test=gender_submission.drop(columns=['PassengerId'])
y_test


# In[41]:


#using svc, getting 100% acc on test set but low on training set
from sklearn.svm import SVC
classifier=SVC(kernel='linear',random_state=42)
classifier.fit(X_train,y_train.values.ravel())
y_pred = classifier.predict(X_test)


# In[45]:


print("Accuracy for training set-",classifier.score(X_train,y_train)*100,"%")
print("Accuracy for test set-",classifier.score(X_test,y_test)*100,"%")


# In[46]:


#csv file
kaggle_data = pd.DataFrame({'PassengerId':gender_submission.PassengerId, 'Survived':y_pred}).set_index('PassengerId')
kaggle_data.to_csv('submission_file.csv')


# In[ ]:




