#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
df=pd.read_csv("Desktop/Shreya/Project/6. Train_naive bayes/Train.csv")
x=df['review']
y=df['label']
df


# In[2]:


import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
sw=(stopwords.words('english'))
ps=PorterStemmer()
import string
punctuations=list(string.punctuation)
sw=sw+punctuations


# In[3]:


def clean_text(sample):
    sample=sample.lower()
    sample=sample.replace("<br /><br />","")
    sample=re.sub("[^a-zA-Z]"," ",sample)                      #if not a character of englishalphabet like not from a-z range replace it with space.We have + sign because we need to keep on cheking till the time we encounter any character which is not in english alphabet.
    
    sample=sample.split()   #splitting sample on the basis of space
    sample=[ps.stem(s) for s in sample if s not in sw]   #List comprehension : if s that is samples element is not in sw then only retain it in the sample and ps helps to reduce the word like helping can be converted to help.
    sample=" ".join(sample)     #join in string format again by space join
    
    return sample


# In[4]:


#clean_text(x[11])

df["cleaned_review"]=x.apply(clean_text)  #Apply clean_text on all the data points present in sample(specifically in review column that is x)


# In[5]:


df     #cleaned_review is advanced version of review


# In[6]:


x=df["cleaned_review"]
print(x)
y


# In[7]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1)


# # naive bayes

# In[8]:


from sklearn.feature_extraction.text import TfidfVectorizer  #vectorize that is marks important words from text only based on repitition and value
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline   #To make important words flow

model=make_pipeline(TfidfVectorizer(), MultinomialNB())

model.fit(x_train,y_train)

labels=model.predict(x_test)


# In[9]:


print(labels)


# In[10]:


from sklearn.metrics import confusion_matrix
mat=confusion_matrix(y_test,labels)
print(mat)
s=(y_test==labels).sum()
acc=s/len(x_test)
acc


# In[11]:


data=pd.read_csv("Desktop/Project/6. Test_naive bayes/Test.csv")
xtest=data['review']


# In[12]:


data["cleaned_review"]=data["review"].apply(clean_text)


# In[13]:


xtest=data["cleaned_review"]


# In[14]:


xtest.shape


# In[15]:


ytest=model.predict(xtest)


# In[16]:


file=pd.DataFrame(data=ytest,columns=["label"])
file.to_csv("y_prediction_naiveBayes.csv",index=True)


# # neural network

# In[17]:


y.value_counts()


# In[18]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[19]:


y=le.fit_transform(y)


# In[20]:


y[:100]


# In[21]:


y.shape


# In[22]:


corpus=df["cleaned_review"].values


# In[23]:


from sklearn.feature_extraction.text import CountVectorizer


# In[24]:


cv=CountVectorizer(max_df=0.5,max_features=50000)   #maxdf is max frequency of a no and we are discarding all the words with max frequency of 0.5 


# In[25]:


x=cv.fit_transform(corpus)


# In[26]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[27]:


tfidf=TfidfTransformer()


# In[28]:


x=tfidf.fit_transform(x)


# In[29]:


print(x)


# In[30]:


x.shape


# In[31]:


y.shape


# In[32]:


from keras import models
from keras.layers import Dense


# In[33]:


model=models.Sequential()
model.add(Dense(16,activation="relu",input_shape=(x.shape[1],)))
model.add(Dense(16,activation="relu"))
model.add(Dense(1,activation="sigmoid"))


# In[34]:


model.summary()


# In[35]:


model.compile(optimizer='rmsprop',loss="binary_crossentropy",metrics=["accuracy"])


# ## spliting data to train and validation

# In[36]:


nx_val=x[:5000].toarray()
nx_train=x[5000:].toarray()
ny_val=y[:5000]
ny_train=y[5000:]


# In[37]:


nx_val.shape, nx_train


# In[38]:


# hist=model.fit(nx_train,ny_train,batch_size=128,epochs=5,validation_data=(nx_val,ny_val))


# In[39]:


# result=hist.history


# In[40]:


# import matplotlib.pyplot as plt
# plt.plot(result['val_accuracy'],label="Val acc")
# plt.plot(result["accuracy"],label="Train acc")
# plt.legend()
# plt.show()


# In[41]:


# plt.plot(result['val_loss'],label="Val loss")
# plt.plot(result["loss"],label="Train loss")
# plt.legend()
# plt.show()


# ## we can see that only till 2 epoch validation's accuracy inncreases so we should train only till 2 epochs rather than 5 as in 5 you are overfiting

# In[42]:


# model.evaluate(nx_val,ny_val)


# In[43]:


hist=model.fit(nx_train,ny_train,batch_size=128,epochs=2,validation_data=(nx_val,ny_val))


# In[44]:


result=hist.history


# In[45]:


import matplotlib.pyplot as plt
plt.plot(result['val_accuracy'],label="Val acc")
plt.plot(result["accuracy"],label="Train acc")
plt.legend()
plt.show()


# In[46]:


plt.plot(result['val_loss'],label="Val loss")
plt.plot(result["loss"],label="Train loss")
plt.legend()
plt.show()


# In[47]:


model.evaluate(nx_val,ny_val)


# In[48]:


nxtest=data["cleaned_review"]
print(nxtest)


# In[49]:


nxtest=cv.transform(nxtest)               #fit is not used as we are not learning foor test data


# In[50]:


nxtest.shape


# In[51]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[52]:


nxtest=tfidf.transform(nxtest)


# In[53]:


nxtest.shape


# In[54]:


# xtest=CountVectorizer(max_df=0.5,max_features=50000)


# In[55]:


# corpus1=data["cleaned_review"].values


# In[56]:


# xtest=cv.fit_transform(corpus1)


# In[62]:


nxtest.sort_indices()
(nxtest)


# In[63]:


# import tensorflow as tf
# tf.sparse.reorder(nxtest)
ny_pred=model.predict(nxtest)


# In[64]:


ny_pred


# # Those values of y whose value is greater than 0.5 , make that values equal to 1
# 

# In[65]:


ny_pred[ny_pred>=0.5]=1


# In[66]:


ny_pred=ny_pred.astype('int')   #convert to int


# In[68]:


dict={0:'neg',1:'pos'}
ny_pred=[dict[p[0]] for p in ny_pred]


# In[69]:


ny_pred


# In[70]:


ids=np.arange(10000)


# In[71]:


final_matrix=np.stack((ids,ny_pred),axis=1)


# In[72]:


final_matrix


# In[73]:


submitfile=pd.DataFrame(final_matrix,columns=["Id","label"])


# In[74]:


submitfile.to_csv("y_predhybridneural.csv",index=False)

