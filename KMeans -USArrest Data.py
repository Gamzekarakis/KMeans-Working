#!/usr/bin/env python
# coding: utf-8

# In[1]:


from warnings import filterwarnings
filterwarnings('ignore')

import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.cluster import KMeans


# ## KMeans

# In[2]:


# In this data, we try to classify US states according to crime types.


# In[8]:


df=pd.read_csv("USArrests.csv")
df.head()


# ### Variables
# 

# * Murder: Number of attacks
# * Assault : Attacks
# * UrbanPop : Population
# * Rape: Assault

# In[10]:


df.index


# In[11]:


# there is no null values
df.isnull().sum()


# In[12]:


# we can see data information like type , count
df.info()


# In[13]:


# Numeric values statistical conclusion
df.describe().T


# In[16]:


# histogram graph of features 
df.hist(figsize=(10,10),color="orange");


# In[17]:



from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=4)
kmeans


# In[18]:


k_fit=kmeans.fit(df)


# In[19]:


k_fit.n_clusters


# In[20]:


k_fit.cluster_centers_


# In[22]:


k_fit.labels_


# In[23]:


# Visualization


# In[25]:


#Try to reduce 4 cluster to 2 cluster

kmeans=KMeans(n_clusters=2)
k_fit=kmeans.fit(df)


# In[26]:


kümeler=k_fit.labels_


# In[31]:


plt.scatter(df.iloc[:,0],df.iloc[:,1],c=kümeler,s=50,cmap='viridis')
merkezler=k_fit.cluster_centers_
plt.scatter(merkezler[:,0],merkezler[:,1],c="black",s=200,alpha=0.5)


# In[32]:


from mpl_toolkits.mplot3d import Axes3D


# In[34]:


#And try 3 cluster and visualization with 3D
kmeans=KMeans(n_clusters=3)
k_fit=kmeans.fit(df)
kümeler=k_fit.labels_
merkezler=kmeans.cluster_centers_


# In[35]:


plt.rcParams['figure.figsize']=(20,15)
fig=plt.figure()
ax=Axes3D(fig)
ax.scatter(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2]);


# In[37]:


plt.rcParams['figure.figsize']=(20,15)
fig=plt.figure()
ax=Axes3D(fig)
ax.scatter(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2],c=kümeler)
ax.scatter(merkezler[:,0],merkezler[:,1],merkezler[:,2],marker='*',
          c='#050505',
          s=1000);


# In[38]:


#States and number of cluster
kmeans=KMeans(n_clusters=3)
k_fit=kmeans.fit(df)
kümeler=k_fit.labels_


# In[39]:


pd.DataFrame({"Eyaletler" : df.index ,"Kümeler" :kümeler})[0:10]


# In[40]:


df["küme_no"]=kümeler


# In[42]:


df.head()


# In[45]:


df["küme_no"]=df["küme_no"]+1


# In[46]:


df.head()


# In[54]:


df["küme_no"].value_counts()


# In[ ]:





# In[ ]:




