#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import plot

print(__doc__)


# In[7]:


ds = pd.read_csv(r"C:\Users\kiree\OneDrive\Desktop\data.csv")


# In[ ]:





# In[14]:


ds.head()


# In[15]:


ds.info()


# In[16]:


ds = ds.dropna()
ds = ds.reset_index(drop=True)
ds.describe()


# In[17]:


ds.corr()


# In[18]:


plt.figure()
sns.heatmap(ds.corr(), cmap='coolwarm')
plt.show()


# In[ ]:





# In[26]:


ds


# In[38]:


ds.shape


# In[27]:


ds.columns
#specify the column names to be modelled
to_model_columns=ds.columns[1:34]
from sklearn.ensemble import IsolationForest
clf=IsolationForest(n_estimators=100, max_samples='auto', contamination=float(.12),                         max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)
clf.fit(ds[to_model_columns])
pred = clf.predict(ds[to_model_columns])
ds['anomaly']=pred
outliers=ds.loc[ds['anomaly']==-1]
outlier_index=list(outliers.index)
#print(outlier_index)
#Find the number of anomalies and normal points here points classified -1 are anomalous
print(ds['anomaly'].value_counts())


# In[35]:


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pca = PCA(n_components=2)  # Reduce to k=2 dimensions
scaler = StandardScaler()
#normalize the metrics
X = scaler.fit_transform(ds[to_model_columns])
X_reduce = pca.fit_transform(X)
fig = plt.figure()


# In[36]:


print(ds.shape)


# In[42]:


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
pca = PCA(2)
pca.fit(ds[to_model_columns])


# In[43]:


res=pd.DataFrame(pca.transform(ds[to_model_columns]))
Z = np.array(res)
plt.title("IsolationForest")
plt.contourf( Z, cmap=plt.cm.Blues_r)
b1 = plt.scatter(res[0], res[1], c='green',
                 s=20,label="normal points")
b1 =plt.scatter(res.iloc[outlier_index,0],res.iloc[outlier_index,1], c='green',s=20,  edgecolor="red",label="predicted outliers")
plt.legend(loc="upper right")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[63]:





# In[ ]:




