#!/usr/bin/env python
# coding: utf-8

# ## Prediction tool to predict PR fraction and severity from echocardiogram.
# In[65]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[42]:


data_pr = pd.read_csv('pr.csv')
data_pr.head()


# In[43]:


#Heat map to vizualize missing values
sns.heatmap(data_pr.isna())


# ### Data Wrangling & Cleanup

# In[44]:


# Filling missing values with mean
COL_LIST = ['ageecho',
 'sex',
 'diagnosis',
 'pulmtype',
 'height',
 'weight',
 'bsa',
 'daysechomri',
 'echopr_recode',
 'plaxvcr',
 'pssavcr',
 'avgvcr',
 'maxvcr',
 'minvcr',
 'prindex',
 'prslope',
 'prpht',
 'pulmvtiratio']
for col in COL_LIST:
    data_pr[col].fillna(data_pr[col].mean(), inplace=True)


# In[45]:


sns.heatmap(data_pr.isna())


# In[46]:


#Converting Yes/No to 1/0
MPA_fr=pd.get_dummies(data_pr['mpa_fr'],drop_first=True)
BPA_fr=pd.get_dummies(data_pr['bpa_fr'],drop_first=True)
data_pr['MPA_FR']=MPA_fr
data_pr['BPA_FR']=BPA_fr
data_pr_final = data_pr.drop(['mpa_fr','bpa_fr'],axis=1)


# In[48]:


# Converting 1,2,3 to '1','2','3' ordered categorical value. 1=Non-Severe,2=Mild-Severe,3=Severe
data_pr_final["mri_pr_cat"] = data_pr_final["mri_pr_cat"].astype('str').astype('object')
cat_type = CategoricalDtype(categories=['1', '2', '3'], ordered=True)
data_pr_final["mri_pr_cat"] = data_pr_final["mri_pr_cat"].astype(cat_type)


# ### The ML section!

# In[53]:


#30% test data, 70% training data
train_df, test_df = train_test_split(data_pr_final, test_size=0.3)
train_df=train_df.sort_index(ascending=True)


# In[54]:


train_df.head()


# In[55]:


# probit
from statsmodels.miscmodels.ordinal_model import OrderedModel
mod_prob = OrderedModel(train_df['mri_pr_cat'],
                        train_df[['ageecho','sex', 'diagnosis',
 'pulmtype',
 'height',
 'weight',
 'bsa',                              
 'daysechomri',
 'echopr_recode',
 'plaxvcr',
 'pssavcr',
 'avgvcr',
 'maxvcr',
 'minvcr',
 'prindex',
 'prslope',
 'prpht',
 'pulmvtiratio',
 'MPA_FR',
 'BPA_FR'
]],
                        distr='probit')


# In[56]:


res_prob = mod_prob.fit(method='bfgs')
res_prob.summary()


# In[370]:


# TODO (Deepa): Need to evaluate feature importance. 
# For now - I am using all features as is.


# In[61]:


# This is prediction on trained dataset to check sanity/quality of fit.
predicted = res_prob.model.predict(res_prob.params, exog=train_df[['ageecho','sex', 'diagnosis',
 'pulmtype',
 'height',
 'weight',
 'bsa',                              
 'daysechomri',
 'echopr_recode',
 'plaxvcr',
 'pssavcr',
 'avgvcr',
 'maxvcr',
 'minvcr',
 'prindex',
 'prslope',
 'prpht',
 'pulmvtiratio',
 'MPA_FR',
 'BPA_FR'
]])
predicted


# In[62]:


# The 6.19902960e-03+3.60027945e-01+6.33773025e-01=1
# Prediction: The category with highest probablity is the winner.


# In[63]:


actual_output = train_df['mri_pr_cat']


# In[64]:


#max probability is predicted output
predicted_output = np.array([])
for row in predicted:
    temp = list(row)
    max_val = max(temp)
    index = temp.index(max_val)
    predicted_output = np.append(predicted_output,[str(index+1)])


# In[374]:


confusion_matrix(actual_output, predicted_output)


# In[66]:


print(classification_report(actual_output, predicted_output))


# In[ ]:



# ## Working on Test Dataset now

# In[67]:


predicted_test = res_prob.model.predict(res_prob.params, exog=test_df[['ageecho','sex', 'diagnosis',
 'pulmtype',
 'height',
 'weight',
 'bsa',                              
 'daysechomri',
 'echopr_recode',
 'plaxvcr',
 'pssavcr',
 'avgvcr',
 'maxvcr',
 'minvcr',
 'prindex',
 'prslope',
 'prpht',
 'pulmvtiratio',
 'MPA_FR',
 'BPA_FR'
]])
predicted_test


# In[69]:


#max probability is predicted output
predicted_output_test = np.array([])
for row in predicted_test:
    temp = list(row)
    max_val = max(temp)
    index = temp.index(max_val)
    predicted_output_test = np.append(predicted_output_test,[str(index+1)])


# In[70]:


actual_output_test = test_df['mri_pr_cat']


# In[73]:


confusion_matrix(actual_output_test, predicted_output_test)


# In[74]:


print(classification_report(actual_output_test, predicted_output_test))

