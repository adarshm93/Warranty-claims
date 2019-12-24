# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 14:31:47 2019

@author: Adarsh
"""
import pandas as pd
import  numpy as np

train=pd.read_csv("C:/Users/Adarsh/Downloads/train.csv")
train=train.drop(["Unnamed: 0"],axis=1)

test=pd.read_csv("C:/Users/Adarsh/Downloads/test_1.csv")
test=test.drop(["Unnamed: 0"],axis=1)
#Replacing UP with Utter Pradesh
train.loc[(train.State=="UP"),"State"]="Uttar Pradesh"
#Replacing claim with Claim
train.loc[(train.Purpose=="claim"),"Purpose"]="Claim"

train.loc[(train.State=="Delhi")|(train.State=="Uttar Pradesh")|
		(train.State=="HP")|(train.State=="J&K"),"Region"]="North"
train.loc[(train.State=="Andhra Pradesh")|(train.State=="Tamilnadu")|(train.State=="Kerala")|(train.State=="Karnataka")|
		(train.State=="Telengana"),"Region"]="South"
train.loc[(train.State=="West Bengal")|(train.State=="Tripura")|(train.State=="Assam")|
		(train.State=="Jharkhand"),"Region"]="East"
train.loc[(train.State=="Gujarat"),"Region"]="West"
train.loc[(train.State=="Bihar")|(train.State=="Haryana")|(train.State=="MP"),"Region"]="North East"
train.loc[(train.State=="Rajasthan"),"Region"]="North West"
train.loc[(train.State=="Odisha"),"Region"]="South East"
train.loc[(train.State=="Maharshtra")|(train.State=="Goa"),"Region"]="South West"
#Seprating Hyderabad from two states 
train.loc[(train.State=="Telengana"),"City"]="Hyderabad 1"

#Replacing UP with Utter Pradesh
test.loc[(test.State=="UP"),"State"]="Uttar Pradesh"
#Replacing claim with Claim
test.loc[(test.Purpose=="claim"),"Purpose"]="Claim"

test.loc[(test.State=="Delhi")|(test.State=="Uttar Pradesh")|
		(test.State=="HP")|(test.State=="J&K"),"Region"]="North"
test.loc[(test.State=="Andhra Pradesh")|(test.State=="Tamilnadu")|(test.State=="Kerala")|(test.State=="Karnataka")|
		(test.State=="Telengana"),"Region"]="South"
test.loc[(test.State=="West Bengal")|(test.State=="Tripura")|(test.State=="Assam")|
		(test.State=="Jharkhand"),"Region"]="East"
test.loc[(test.State=="Gujarat"),"Region"]="West"
test.loc[(test.State=="Bihar")|(test.State=="Haryana")|(test.State=="MP"),"Region"]="North East"
test.loc[(test.State=="Rajasthan"),"Region"]="North West"
test.loc[(test.State=="Odisha"),"Region"]="South East"
test.loc[(test.State=="Maharshtra")|(test.State=="Goa"),"Region"]="South West"
#Seprating Hyderabad from two states 
test.loc[(test.State=="Telengana"),"City"]="Hyderabad 1"

#converting into binary
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
train["Region"]=lb.fit_transform(train["Region"])
train["State"]=lb.fit_transform(train["State"])
train["Area"]=lb.fit_transform(train["Area"])
train["City"]=lb.fit_transform(train["City"])
train["Consumer_profile"]=lb.fit_transform(train["Consumer_profile"])
train["Product_category"]=lb.fit_transform(train["Product_category"])
train["Product_type"]=lb.fit_transform(train["Product_type"])
train["Purchased_from"]=lb.fit_transform(train["Purchased_from"])
train["Purpose"]=lb.fit_transform(train["Purpose"])

test["Region"]=lb.fit_transform(test["Region"])
test["State"]=lb.fit_transform(test["State"])
test["Area"]=lb.fit_transform(test["Area"])
test["City"]=lb.fit_transform(test["City"])
test["Consumer_profile"]=lb.fit_transform(test["Consumer_profile"])
test["Product_category"]=lb.fit_transform(test["Product_category"])
test["Product_type"]=lb.fit_transform(test["Product_type"])
test["Purchased_from"]=lb.fit_transform(test["Purchased_from"])
test["Purpose"]=lb.fit_transform(test["Purpose"])

#Median Imputation
train.fillna(train.median(), inplace=True)
train.isna().sum()

#Median Imputation
test.fillna(test.median(), inplace=True)
test.isna().sum()

# Standardization
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

df_norm = norm_func(train.iloc[:,:19])
df_norm.describe()

train1= pd.concat([df_norm, train.iloc[:,19]], axis = 1)

test1 = norm_func(test.iloc[:,:19])
test1.describe()

from sklearn.utils import resample

# Separate input features and target
x = train1.iloc[:,:19]
y = train1.iloc[:,19]
from sklearn.model_selection import train_test_split
# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.40, random_state=53)

# concatenate our training data back together
X = pd.concat([X_train, y_train], axis=1)

# separate minority and majority classes
not_fraud = X[X.Fraud==0]
fraud = X[X.Fraud==1]

# upsample minority
fraud_upsampled = resample(fraud,
                          replace=True, # sample with replacement
                          n_samples=len(not_fraud), # match number in majority class
                          random_state=53) # reproducible results

# combine majority and upsampled minority
upsampled = pd.concat([not_fraud, fraud_upsampled])

result = upsampled.reset_index() 
r2=result.drop(["index"],axis=1)


############### Feature Selection using Tree Classifier ##############

from sklearn.tree import  ExtraTreeClassifier
a = r2.iloc[:,0:19]  #independent columns
b = r2.iloc[:,-1]    #target column

model = ExtraTreeClassifier()
model.fit(a,b)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=a.columns)
feat_importances.nlargest(19).plot(kind='barh')


####################### Cross Validation ######################

colnames = list(r2.columns)
predictors = colnames[:19]
target = colnames[19]

Xx = r2[predictors]
Yy = r2[target]

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=300, random_state=24)
from sklearn.model_selection import cross_val_score
all_accuracies = cross_val_score(classifier,Xx,Yy,cv=10)
print(all_accuracies)
print(all_accuracies.mean()) #96.13

import catboost as ctb
modell = ctb.CatBoostClassifier()
all_accuraciess = cross_val_score(modell,Xx,Yy,cv=10)
print(all_accuraciess)
print(all_accuraciess.mean())


from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1)
all_accuraciesss = cross_val_score(abc,Xx,Yy,cv=10)
print(all_accuraciesss)
print(all_accuraciesss.mean())

from xgboost import XGBClassifier
model = XGBClassifier()
all_accuracy = cross_val_score(model,Xx,Yy,cv=10)
print(all_accuracy)
print(all_accuracy.mean())

################ Model building ##################

#Random Forest model building with feature selection
r2=r2.drop(["Product_category"],axis=1)
colnames = list(r2.columns)
predictors = colnames[:18]
target = colnames[18]

X = r2[predictors]
Y = r2[target]


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=4,oob_score=True,n_estimators=200,criterion="entropy")

rf.fit(r2[predictors],r2[target])
rf.estimators_ # 
rf.n_classes_ # Number of levels in class labels  

rf.n_outputs_ # Number of outputs when fit performed

rf.oob_score_  
rf.predict(X)


pred_train=rf.predict(r2[predictors])
pred_test = rf.predict(test1[predictors])
pd.Series(pred_test).value_counts()

pd.crosstab(r2[target],pred_train)

# Accuracy = train
np.mean(r2.Fraud == rf.predict(r2[predictors]))#97.84

# Accuracy = Test
#kaggle

