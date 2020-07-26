#!/usr/bin/env python
# coding: utf-8

# In[1]:


import download_utils


# In[5]:


import os
print(os.path)


# In[6]:


import sys
print(sys.path)


# ## Keras resources

# In[ ]:


download_utils.download_all_keras_resources("readonly/keras/models", "readonly/keras/datasets")


# ## Week 3 resources

# In[ ]:


download_utils.download_week_3_resources("readonly/week3")


# ## Week 4 resources

# In[ ]:


download_utils.download_week_4_resources("readonly/week4")


# ## Week 6 resources

# In[ ]:


download_utils.download_week_6_resources("readonly/week6")

