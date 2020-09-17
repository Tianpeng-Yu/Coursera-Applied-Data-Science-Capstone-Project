#!/usr/bin/env python
# coding: utf-8

# # Suggestions on Car Accident Severity In Seattle with Machine Learning Models

# Business Understanding: car accidents have great impact on people's lives. To help reduce the severity and frequency of car collision, I want to use the Seattle car collision data to generate insights on how modeling can help reduce accidents. Given the attributes including location, weather conditions and address type, we can see which factors attributes to car accidents most and how we can alert the driver in advance. 
# 
# This research and report would be beneficial to the local government, people who live in seattle and also car insurance. And later I would use machine learning models to interpret the logics and advice to reduce car accidents and injuries in Seattle.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Load Data

# In[ ]:


#Load data
#!pip install wget
import wget
url = 'https://s3.us.cloud-object-storage.appdomain.cloud/cf-courses-data/CognitiveClass/DP0701EN/version-2/Data-Collisions.csv'
filename = wget.download(url)


# In[3]:


df = pd.read_csv(filename)
df.head(10)


# In[4]:


df.shape


# ### Data Selection

# In[5]:


# Print out the existing attributes and see if there is any valuable features
num_count = ('SEVERITYCODE', 'OBJECTID', 'INCKEY', 'COLDETKEY', 'REPORTNO',
       'STATUS', 'ADDRTYPE', 'INTKEY', 'LOCATION', 'EXCEPTRSNCODE','COLLISIONTYPE',
       'PERSONCOUNT', 'PEDCOUNT', 'PEDCYLCOUNT', 'VEHCOUNT', 'INCDATE',
       'INCDTTM', 'JUNCTIONTYPE', 'SDOT_COLCODE', 'SDOT_COLDESC',
       'INATTENTIONIND', 'UNDERINFL', 'WEATHER', 'ROADCOND', 'LIGHTCOND',
       'PEDROWNOTGRNT', 'SDOTCOLNUM', 'SPEEDING', 'ST_COLCODE', 'ST_COLDESC',
       'SEGLANEKEY', 'CROSSWALKKEY', 'HITPARKEDCAR')

for i in num_count:
    print ('the number of each',':', i, 'is')
    print (df[i].value_counts(dropna = False),'\n')
print('')


# Based on the result, I selected ADDRTYPE, COLLISIONTYPE, WEATHER, ROADCOND, LIGHTCOND as the features to analyze in my model. For INATTENTIONIND and UNDERINFL which means whether the driver was paying attention or using alcohol and for PEDROWNOTGRNT and SPEEDING, they only have positive and NA values. So I will need further data to include these features into the model in the future.

# # Data visualization and pre-processing
# 

# ### First I will use all data incuding the NaN values to fit the models and then compare with the models excluding the NaN
# 

# In[7]:


#Select the target variables and features that will be used
df2 = df[['SEVERITYCODE', 'X', 'Y', 'ADDRTYPE', 'COLLISIONTYPE',
          'WEATHER', 'ROADCOND', 'LIGHTCOND','LOCATION']]
df2.rename(columns={'X':'Lon'}, inplace=True)
df2.rename(columns={'Y':'Lat'}, inplace=True)
df2.head()


# In[9]:


#Take a look at the data types
df2.dtypes


# In[10]:


#Take a look at the labeled data: Severity code
df['SEVERITYCODE'].value_counts(dropna = False)


# In[11]:


#Total number of empty inputs in " ROADCOND"
df['ROADCOND'].value_counts(dropna = False)


# In[12]:


#Total number of empty inputs in "WEATHER"
df['WEATHER'].value_counts(dropna = False)


# In[13]:


# notice: installing seaborn might takes a few minutes
#!conda install -c anaconda seaborn -y


# In[14]:


import seaborn as sns
sns.set_style("whitegrid")
pic1 = sns.countplot(x="ADDRTYPE", hue = 'SEVERITYCODE', data=df2, palette="rocket")
plt.title('The Number of Accidents by Collision Address Type')


# In[15]:


pic2 = sns.countplot(x="WEATHER", hue = 'SEVERITYCODE', data=df2, palette="rocket")
plt.title('The Number of Accidents by Collision Address Type')
plt.xticks(rotation=45)


# In[17]:


#Use Folium to see on the map
get_ipython().system('conda install -c conda-forge folium=0.5.0 --yes')


# In[18]:


import folium

print('Folium installed and imported!')


# In[19]:


df2_acc = df2.groupby(['Lon', 'Lat', 'LOCATION']).size().reset_index(name='Count') 
df2_acc['Frequency'] = df2_acc['Count'].apply(lambda x: 'Often' if x>15  else ('Sometimes' if x > 5 else 'Few'))
df2_acc


# In[20]:


traffic_map = folium.Map(location=[47.608, -122.335],zoom_start = 10)
colordict = {'Few': 'blue', 'Sometimes': 'orange', 'Often': 'red'}


for Lon, Lat, LOCATION, Count, Frequency in zip(df2_acc['Lon'], df2_acc['Lat'], df2_acc['LOCATION'], df2_acc['Count'], df2_acc['Frequency']):
    folium.CircleMarker(
        [Lat, Lon],
        radius=.1*Count,
        popup = ('Street: ' + str(LOCATION).capitalize() + '<br>'
                 'Accident: ' + str(Count) + '<br>'
                 'Accident level: ' + str(Frequency) +'%'
                ),
        color='b',
        key_on = Frequency,
        threshold_scale=[0,1,2,3],
        fill_color=colordict[Frequency],
        fill=True,
        fill_opacity=0.7
        ).add_to(traffic_map)



traffic_map


# # Pre-processing: Feature selection/extraction

# In[21]:


df2.groupby(['WEATHER'])['SEVERITYCODE'].value_counts(normalize=True)


# ## Convert Categorical features to numerical values

# ### One Hot Encoding

# In[62]:


encoding_ADDRTYPE = {'ADDRTYPE': {'Block':2, 'Intersection':3, 'Alley':4, np.nan: 0} }
df2.replace(encoding_ADDRTYPE, inplace=True)
df2['ADDRTYPE'].value_counts()


# In[63]:


encoding_COLLISIONTYPE = {'COLLISIONTYPE': 
                          {'Parked Car':2, 
                           'Angles':3, 
                           'Rear Ended':4, 
                           'Other':1,
                           'Sideswipe':5,
                           'Left Turn':6, 
                           'Pedestrian':7, 
                           'Cycles':8, 
                            np.NaN :0, 
                           'Right Turn':9, 
                           'Head On':10} }
df2.replace(encoding_COLLISIONTYPE, inplace=True)
df2['COLLISIONTYPE'].value_counts()


# In[66]:


encoding_WEATHER = {'WEATHER': 
                    {np.nan:0,
                     'Unknown':1,
                     'Other':1,
                     'Clear':2, 
                     'Raining':3, 
                     'Overcast':4, 
                     'Snowing':5, 
                     'Fog/Smog/Smoke':6, 
                     'Sleet/Hail/Freezing Rain':7, 
                     'Blowing Sand/Dirt':8, 
                     'Severe Crosswind':9,
                     'Partly Cloudy':10}}

df2.replace(encoding_WEATHER, inplace=True)
df2['WEATHER'].value_counts()


# In[70]:


encoding_ROADCOND = {'ROADCOND': 
                     {np.nan:0,
                      'Unknown':1,
                      'Other':1,
                      'Dry':2, 
                      'Wet':3, 
                      'Ice':4,  
                      'Snow/Slush':5, 
                      'Standing Water' :6, 
                      'Sand/Mud/Dirt':7,
                      'Oil':8} }

df2.replace(encoding_ROADCOND, inplace=True)
df2['ROADCOND'].value_counts()


# In[68]:


encoding_LIGHTCOND = {'LIGHTCOND': 
                     {np.nan:0,
                      'Unknown':1,
                      'Other':1,
                      'Daylight':2, 
                      'Dark - Street Lights On':3, 
                      'Dusk':4,  
                      'Dawn':5, 
                      'Dark - No Street Lights' :6, 
                      'Dark - Street Lights Off':7,
                      'Dark - Unknown Lighting':1} }

df2.replace(encoding_LIGHTCOND, inplace=True)
df2['LIGHTCOND'].value_counts()


# In[73]:


Feature = df2[['ADDRTYPE', 'COLLISIONTYPE', 'WEATHER', 'ROADCOND', 'LIGHTCOND']]
Feature.head()


# In[74]:


X = Feature
Y = df2['SEVERITYCODE'].values


# In[75]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# # K Nearest Neighbor(KNN)

# In[77]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix  

Ks = 12
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(3,Ks):
    
    #Train Model and Predict  
    knn = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=knn.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc


# In[78]:


#Plot to find the best k
plt.plot(range(1,Ks),mean_acc)
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 

knn_best = KNeighborsClassifier(n_neighbors=mean_acc.argmax()+1).fit(X_train, y_train)


# In[88]:


# fit the model with data (occurs in-place)
knn_best.fit(X_train, y_train)

y_pred = knn_best.predict(X_test)

print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))


# In[92]:


knn_best.score(X_test, y_test)


# In[93]:


knn_best.score(X_train, y_train)


# ### LogisticRegression

# In[94]:


from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(C=0.01, solver="liblinear").fit(X_train,y_train)
LR

yhat_lr = LR.predict(X_test)
yhat_lr

print(confusion_matrix(y_test, yhat_lr))  
print(classification_report(y_test, yhat_lr))


# In[96]:


LR.score(X_test, y_test)


# In[97]:


LR.score(X_train, y_train)


# ## The data is underfit, so I created another dataframe without NA values

# In[8]:


#Create another dataframe without NA values to compare later
df2_dropna = df2.dropna()
df2_dropna
df2_dropna.head()


# In[9]:


#Balance the data
from sklearn.utils import resample

df2_dropna_major = df2_dropna[df2_dropna.SEVERITYCODE == 1]
df2_dropna_minor = df2_dropna[df2_dropna.SEVERITYCODE == 2]

df2_dropna_major_desample = resample(df2_dropna_major, replace = False, n_samples = 56000, random_state = 123)
balanced_df2 = pd.concat([df2_dropna_major_desample,df2_dropna_minor])
balanced_df2['SEVERITYCODE'].value_counts()


# ## Visualization for DF_DROPNA

# In[10]:


import seaborn as sns
sns.set_style("whitegrid")
pic_dropna1 = sns.countplot(x="ADDRTYPE", hue = 'SEVERITYCODE', data=balanced_df2, palette="rocket")
plt.title('The Number of Accidents by Collision Address Type')


# In[11]:


pic_dropna2 = sns.countplot(x="WEATHER", hue = 'SEVERITYCODE', data=balanced_df2, palette="rocket")
plt.title('The Number of Accidents by Collision Address Type')
plt.xticks(rotation=45)


# In[12]:


pic_dropna3 = sns.countplot(x="ROADCOND", hue = 'SEVERITYCODE', data=balanced_df2, palette="rocket")
plt.title('The Number of Accidents by Road Condition')
plt.xticks(rotation=45)


# In[13]:


#!conda install -c conda-forge folium=0.5.0 --yes
import folium
print('Folium installed and imported!')


# In[14]:


df2_acc = balanced_df2.groupby(['Lon', 'Lat', 'LOCATION']).size().reset_index(name='Count') 
df2_acc['Frequency'] = df2_acc['Count'].apply(lambda x: 'Often' if x>15  else ('Sometimes' if x > 5 else 'Few'))
df2_acc


# In[43]:


traffic_map = folium.Map(location=[47.608, -122.335],zoom_start = 10)
colordict = {'Few': 'blue', 'Sometimes': 'orange', 'Often': 'red'}


for Lon, Lat, LOCATION, Count, Frequency in zip(df2_acc['Lon'], df2_acc['Lat'], df2_acc['LOCATION'], df2_acc['Count'], df2_acc['Frequency']):
    folium.CircleMarker(
        [Lat, Lon],
        radius=.1*Count,
        popup = ('Street: ' + str(LOCATION).capitalize() + '<br>'
                 'Accident: ' + str(Count) + '<br>'
                 'Accident level: ' + str(Frequency) +'%'
                ),
        color='b',
        key_on = Frequency,
        threshold_scale=[0,1,2,3],
        fill_color=colordict[Frequency],
        fill=True,
        fill_opacity=0.7
        ).add_to(traffic_map)



traffic_map


# In[15]:


encoding_ADDRTYPE = {'ADDRTYPE': {'Block':2, 'Intersection':3} }
df2_dropna.replace(encoding_ADDRTYPE, inplace=True)
df2_dropna['ADDRTYPE'].value_counts()


# In[16]:


encoding_COLLISIONTYPE = {'COLLISIONTYPE': 
                          {'Parked Car':2, 
                           'Angles':3, 
                           'Rear Ended':4, 
                           'Other':1,
                           'Sideswipe':5,
                           'Left Turn':6, 
                           'Pedestrian':7, 
                           'Cycles':8, 
                           'Right Turn':9, 
                           'Head On':10} }
df2_dropna.replace(encoding_COLLISIONTYPE, inplace=True)
df2_dropna['COLLISIONTYPE'].value_counts()


# In[17]:


encoding_WEATHER = {'WEATHER': 
                    {'Unknown':1,
                     'Other':1,
                     'Clear':2, 
                     'Raining':3, 
                     'Overcast':4, 
                     'Snowing':5, 
                     'Fog/Smog/Smoke':6, 
                     'Sleet/Hail/Freezing Rain':7, 
                     'Blowing Sand/Dirt':8, 
                     'Severe Crosswind':9,
                     'Partly Cloudy':10}}

df2_dropna.replace(encoding_WEATHER, inplace=True)
df2_dropna['WEATHER'].value_counts()


# In[18]:


encoding_ROADCOND = {'ROADCOND': 
                     {'Unknown':1,
                      'Other':1,
                      'Dry':2, 
                      'Wet':3, 
                      'Ice':4,  
                      'Snow/Slush':5, 
                      'Standing Water' :6, 
                      'Sand/Mud/Dirt':7,
                      'Oil':8} }

df2_dropna.replace(encoding_ROADCOND, inplace=True)
df2_dropna['ROADCOND'].value_counts()


# In[19]:


encoding_LIGHTCOND = {'LIGHTCOND': 
                     {'Unknown':1,
                      'Other':1,
                      'Daylight':2, 
                      'Dark - Street Lights On':3, 
                      'Dusk':4,  
                      'Dawn':5, 
                      'Dark - No Street Lights' :6, 
                      'Dark - Street Lights Off':7,
                      'Dark - Unknown Lighting':1} }

df2_dropna.replace(encoding_LIGHTCOND, inplace=True)
df2_dropna['LIGHTCOND'].value_counts()


# In[20]:


df2_dropna['SEVERITYCODE'].value_counts()


# In[21]:


#Balance the data
from sklearn.utils import resample

df2_dropna_major = df2_dropna[df2_dropna.SEVERITYCODE == 1]
df2_dropna_minor = df2_dropna[df2_dropna.SEVERITYCODE == 2]

df2_dropna_major_desample = resample(df2_dropna_major, replace = False, n_samples = 56000, random_state = 123)
balanced_df2 = pd.concat([df2_dropna_major_desample,df2_dropna_minor])
balanced_df2['SEVERITYCODE'].value_counts()

Feature = balanced_df2[['ADDRTYPE', 'COLLISIONTYPE', 'WEATHER', 'ROADCOND', 'LIGHTCOND']]
Feature.head()


# In[22]:


X = Feature
Y = balanced_df2['SEVERITYCODE'].values
print (X.shape)
print (Y.shape)


# In[23]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# # KNN

# In[24]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix  

Ks = 12
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(3,Ks):
    
    #Train Model and Predict  
    knn = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=knn.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc


# In[25]:


#Plot to find the best k
plt.plot(range(1,Ks),mean_acc)
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 

knn_best = KNeighborsClassifier(n_neighbors=mean_acc.argmax()+1).fit(X_train, y_train)


# In[26]:


# fit the model with data (occurs in-place)
knn_best.fit(X_train, y_train)

y_pred = knn_best.predict(X_test)

print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))


# In[121]:


knn_best.score(X_test, y_test)


# In[122]:


knn_best.score(X_train, y_train)


# In[124]:


from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
jc1 = jaccard_similarity_score(y_test, y_pred)
fs1 = f1_score(y_test, y_pred, average='weighted')
print(jc1,fs1)


# In[36]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

knn_best = KNeighborsClassifier(n_neighbors=mean_acc.argmax()+1).fit(X_train, y_train)
print("Test set score: {:.4f}".format(knn_best.score(X_test, y_test)))

y_scores = knn_best.predict_proba(X_test)
fpr, tpr, threshold = roc_curve(y_test, y_scores[:, 1], pos_label= 2)
roc_auc = auc(fpr, tpr)
print(roc_auc)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of kNN')
plt.show()


# # SVM

# In[28]:


from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
import pylab as pl
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc

n_samples, n_features = X.shape
random_state = np.random.RandomState(0)
# shuffle and split training and test sets
X, Y = shuffle(X, Y, random_state=random_state)
half = int(n_samples / 2)
X_train, X_test = X[:half], X[half:]
y_train, y_test = Y[:half], Y[half:]

classifier = svm.SVC(kernel='linear', probability=True)
SVC_probas = classifier.fit(X_train, y_train).predict_proba(X_test)

# Compute ROC curve and area the curve
fpr, tpr, thresholds = roc_curve(y_test, SVC_probas[:, 1],pos_label=2)
roc_auc = auc(fpr, tpr)
print ("Area under the ROC curve : %f" % roc_auc)


# In[33]:


pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('ROC Curve of SVM')
pl.legend(loc="lower right")
pl.show()


# In[31]:


y_predsvm = classifier.predict(X_train)
print(confusion_matrix(y_test, y_predsvm))  
print(classification_report(y_test, y_predsvm))


# # Logistic Regression

# In[18]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix  

LR = LogisticRegression()
LR.fit(X_train, y_train)
y_predlr = LR.predict(X_test)
print(confusion_matrix(y_test, y_predlr))  
print(classification_report(y_test, y_predlr))


# In[23]:


from sklearn.metrics import roc_curve, auc
y_predlr_probas = LR.predict_proba(X_test)[:,1]

# Compute ROC curve and area the curve
fpr, tpr, thresholds = roc_curve(y_test, y_predlr_probas, pos_label=2)
roc_auc = auc(fpr, tpr)
print ("Area under the ROC curve : %f" % roc_auc)


# In[32]:


plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve of Logistic Regression')
plt.legend(loc="lower right")
plt.show()


# In[26]:


df2_dropna.shape


# In[27]:


194673-184146


# In[ ]:




