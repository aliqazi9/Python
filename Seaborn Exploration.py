#!/usr/bin/env python
# coding: utf-8

# In[7]:


import seaborn as sns
import pandas as pd
import numpy as np


# In[2]:


sns.get_dataset_names()


# In[4]:


tips = sns.load_dataset("tips")
iris = sns.load_dataset("iris")
titanic = sns.load_dataset("titanic")
planets = sns.load_dataset("planets")


# In[5]:


tips


# In[11]:


sns.scatterplot(x="tip" , y="total_bill", data=tips, hue="day", size="size", palette="YlGnBu")


# In[15]:


sns.histplot(tips['tip'], kde=True, bins=15)


# In[16]:


sns.displot(tips['tip'], kde=True, bins=15)


# In[22]:


sns.barplot(x="sex", y="tip", data=tips, palette="YlGnBu")


# In[27]:


sns.boxplot(x="day", y="tip", data=tips, hue="sex", palette="YlGnBu")


# In[28]:


sns.boxplot(x="day", y="total_bill", data=tips, hue="sex", palette="YlGnBu")


# In[32]:


sns.stripplot(x="day", y="tip", data=tips, hue="sex", palette="YlGnBu", dodge=True)


# In[33]:


sns.stripplot(x="day", y="tip", data=tips, hue="sex", palette="YlGnBu")


# In[35]:


sns.jointplot(x="tip" , y="total_bill", data=tips)


# In[36]:


sns.jointplot(x="tip" , y="total_bill", data=tips, kind="reg")


# In[37]:


sns.jointplot(x="tip" , y="total_bill", data=tips, kind="kde")


# In[38]:


sns.jointplot(x="tip" , y="total_bill", data=tips, kind="kde", shade=True)


# In[41]:


sns.jointplot(x="tip" , y="total_bill", data=tips, kind="kde", shade=True, cmap="YlGnBu")


# In[43]:


sns.jointplot(x="tip" , y="total_bill", data=tips, kind="hex", cmap="YlGnBu")


# In[44]:


titanic 


# In[47]:


sns.pairplot(titanic.select_dtypes(['number']), hue="pclass")


# In[48]:


titanic.corr()


# In[50]:


sns.heatmap(titanic.corr(), annot=True, cmap="coolwarm")


# In[51]:


iris


# In[52]:


sns.clustermap(iris.drop("species", axis=1))

