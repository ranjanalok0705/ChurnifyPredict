#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries

# In[ ]:





# In[49]:


import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from imblearn.combine import SMOTEENN


# #### Reading csv

# In[50]:


df=pd.read_csv("tel_churn.csv")
df.head()


# In[51]:


df=df.drop('Unnamed: 0',axis=1)


# In[52]:


x=df.drop('Churn',axis=1)
x


# In[53]:


y=df['Churn']
y


# ##### Train Test Split

# In[54]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# #### Decision Tree Classifier

# In[55]:


model_dt=DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=6, min_samples_leaf=8)


# In[56]:


model_dt.fit(x_train,y_train)


# In[57]:


y_pred=model_dt.predict(x_test)
y_pred


# In[58]:


model_dt.score(x_test,y_test)


# In[59]:


print(classification_report(y_test, y_pred, labels=[0,1]))


# ###### As you can see that the accuracy is quite low, and as it's an imbalanced dataset, we shouldn't consider Accuracy as our metrics to measure the model, as Accuracy is cursed in imbalanced datasets.
# 
# ###### Hence, we need to check recall, precision & f1 score for the minority class, and it's quite evident that the precision, recall & f1 score is too low for Class 1, i.e. churned customers.
# 
# ###### Hence, moving ahead to call SMOTEENN (UpSampling + ENN)

# In[63]:


sm = SMOTEENN()
X_resampled, y_resampled = sm.fit_resample(x,y)


# In[64]:


xr_train,xr_test,yr_train,yr_test=train_test_split(X_resampled, y_resampled,test_size=0.2)


# In[65]:


model_dt_smote=DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=6, min_samples_leaf=8)


# In[66]:


model_dt_smote.fit(xr_train,yr_train)
yr_predict = model_dt_smote.predict(xr_test)
model_score_r = model_dt_smote.score(xr_test, yr_test)
print(model_score_r)
print(metrics.classification_report(yr_test, yr_predict))


# In[67]:


print(metrics.confusion_matrix(yr_test, yr_predict))


# ###### Now we can see quite better results, i.e. Accuracy: 92 %, and a very good recall, precision & f1 score for minority class.
# 
# ###### Let's try with some other classifier.

# #### Random Forest Classifier

# In[68]:


from sklearn.ensemble import RandomForestClassifier


# In[69]:


model_rf=RandomForestClassifier(n_estimators=100, criterion='gini', random_state = 100,max_depth=6, min_samples_leaf=8)


# In[70]:


model_rf.fit(x_train,y_train)


# In[71]:


y_pred=model_rf.predict(x_test)


# In[72]:


model_rf.score(x_test,y_test)


# In[73]:


print(classification_report(y_test, y_pred, labels=[0,1]))


# In[ ]:





# In[ ]:





# In[77]:


sm = SMOTEENN()
X_resampled1, y_resampled1 = sm.fit_resample(x,y)


# In[79]:


xr_train1,xr_test1,yr_train1,yr_test1=train_test_split(X_resampled1, y_resampled1,test_size=0.2)


# In[80]:


model_rf_smote=RandomForestClassifier(n_estimators=100, criterion='gini', random_state = 100,max_depth=6, min_samples_leaf=8)


# In[81]:


model_rf_smote.fit(xr_train1,yr_train1)


# In[82]:


yr_predict1 = model_rf_smote.predict(xr_test1)


# In[83]:


model_score_r1 = model_rf_smote.score(xr_test1, yr_test1)


# In[92]:


print(model_score_r1)
print(metrics.classification_report(yr_test1, yr_predict1))


# #### Pickling the model

# In[93]:


import pickle


# In[94]:


filename = 'model.sav'


# In[95]:


pickle.dump(model_rf_smote, open(filename, 'wb'))


# In[96]:


load_model = pickle.load(open(filename, 'rb'))


# In[97]:


model_score_r1 = load_model.score(xr_test1, yr_test1)


# In[98]:


model_score_r1


# In[ ]:





# ##### Our final model i.e. RF Classifier with SMOTEENN, is now ready and dumped in model.sav, which we will use and prepare API's so that we can access our model from UI.
