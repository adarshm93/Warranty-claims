

###########################################
############## Importing Packages #########
###########################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import seaborn as sn
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.metrics import  recall_score
from sklearn.metrics import  precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
import catboost as ctb
from sklearn.model_selection import cross_val_score
from sklearn.tree import  ExtraTreeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeClassifier
from keras.utils import to_categorical



##############################################################
######################## Importing Dataset ##################
##############################################################


claim=pd.read_csv("E:/ADM/Excelr solutions/Warranty  claims project/Claims (1).csv")

#Checking Dataset dimension,info
claim.columns
claim.info()
claim.head()
claim.isnull().sum()
claim=claim.drop(["Unnamed: 0"],axis=1)
claim.State.value_counts()
claim.Fraud.value_counts() 
claim.Region.value_counts()


################################################################
##################### Data Cleansing ###########################
################################################################


#Replacing UP with Utter Pradesh
claim.loc[(claim.State=="UP"),"State"]="Uttar Pradesh"
#Replacing claim with Claim
claim.loc[(claim.Purpose=="claim"),"Purpose"]="Claim"

claim.loc[(claim.State=="Delhi")|(claim.State=="Uttar Pradesh")|
		(claim.State=="HP")|(claim.State=="J&K"),"Region"]="North"
claim.loc[(claim.State=="Andhra Pradesh")|(claim.State=="Tamilnadu")|(claim.State=="Kerala")|(claim.State=="Karnataka")|
		(claim.State=="Telengana"),"Region"]="South"
claim.loc[(claim.State=="West Bengal")|(claim.State=="Tripura")|(claim.State=="Assam")|
		(claim.State=="Jharkhand"),"Region"]="East"
claim.loc[(claim.State=="Gujarat"),"Region"]="West"
claim.loc[(claim.State=="Bihar")|(claim.State=="Haryana")|(claim.State=="MP"),"Region"]="North East"
claim.loc[(claim.State=="Rajasthan"),"Region"]="North West"
claim.loc[(claim.State=="Odisha"),"Region"]="South East"
claim.loc[(claim.State=="Maharshtra")|(claim.State=="Goa"),"Region"]="South West"
#Seprating Hyderabad from two states 
claim.loc[(claim.State=="Telengana"),"City"]="Hyderabad 1"


########################################################
########################### EDA ########################
########################################################


#Identify duplicates records in the data
dupes=claim.duplicated()
sum(dupes)
#Removing Duplicates
claim1=claim.drop_duplicates() ######

claim1.Fraud.value_counts()
claim1.shape
describe= claim1.describe() #mean,Std. dev, range
claim1.describe().T
claim1.median()
y=claim1.mode()
claim1.var()
claim_range=max(claim1.Product_Age)-min(claim1.Product_Age)
claim1.skew()

#Finding outliers with index
def detect_outliers(x):
   q1 = np.percentile(x,25)
   q3 = np.percentile(x,75)
   iqr = q3 - q1
   lower = q1-(1.5*iqr)
   upper = q3+(1.5*iqr)
   outlier_indices = list(x.index[(x<lower) | (x>upper)])
   outlier_value = list(x[outlier_indices])
   
   return outlier_indices, outlier_value
indics, values =  detect_outliers(claim1["Call_details"])
print(indics)
print(values)
indics, values =  detect_outliers(claim1["Product_Age"])
print(indics)
print(values)


#histograms for each variable in df
hist =claim1.hist(bins=20,figsize =(14,14))

sns.countplot(data = claim1, x = 'Region')
sns.countplot(data = claim1, x = 'State')
sns.countplot(data = claim1, x = 'Area')
sns.countplot(data = claim1, x = 'City')
sns.countplot(data = claim1, x = 'Consumer_profile')
sns.countplot(data = claim1, x = 'Product_category')
sns.countplot(data = claim1, x = 'Product_type')
sns.countplot(data = claim1, x = 'Claim_Value')
sns.countplot(data = claim1, x = 'Service_Centre')
sns.countplot(data = claim1, x = 'Product_Age')
sns.countplot(data = claim1, x = 'Purchased_from')
sns.countplot(data = claim1, x = 'Call_details')
sns.countplot(data = claim1, x = 'Purpose')
sns.countplot(data = claim1, x = 'Fraud')

#create a boxplot for every column in df
boxplot = claim1.boxplot(grid=True, vert=True,fontsize=13)

#create the correlation matrix heat map
plt.figure(figsize=(18,16))
sns.heatmap(claim1.corr(),linewidths=.6,cmap="YlGnBu", annot=True)
plt.yticks(rotation=0)

#pair plots
g = sns.pairplot(claim1)

# Plotting a scatter plot
fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(claim1['Fraud'], claim1['Claim_Value'])
ax.set_xlabel('Fraud')
ax.set_ylabel('Claim_Value')
plt.show()

sb.factorplot(x='Fraud' ,col='Region' ,kind='count', data=claim1);
sb.factorplot(x='Fraud' ,col='State' ,kind='count', data=claim1);
sb.factorplot(x='Fraud' ,col='Claim_Value' ,kind='count', data=claim1);
sb.factorplot(x='Fraud' ,col='City' ,kind='count', data=claim1);
sb.factorplot(x='Fraud' ,col='Consumer_profile' ,kind='count', data=claim1);
sb.factorplot(x='Fraud' ,col='Product_type' ,kind='count', data=claim1);
sb.factorplot(x='Fraud' ,col='Purchased_from' ,kind='count', data=claim1);
sb.factorplot(x='Fraud' ,col='Purpose' ,kind='count', data=claim1);

sb.factorplot(x='Fraud' ,col='AC_1001_Issue' ,kind='count', data=claim1);
sb.factorplot(x='Fraud' ,col='AC_1002_Issue' ,kind='count', data=claim1);
sb.factorplot(x='Fraud' ,col='AC_1003_Issue' ,kind='count', data=claim1);

sb.factorplot(x='Fraud' ,col='TV_2001_Issue' ,kind='count', data=claim1);
sb.factorplot(x='Fraud' ,col='TV_2002_Issue' ,kind='count', data=claim1);
sb.factorplot(x='Fraud' ,col='TV_2003_Issue' ,kind='count', data=claim1);

pd.crosstab(claim1['City'],claim1['Fraud']).plot(kind = 'bar' )

## Scatter plots
plt.figure(figsize = (15,9))
sn.regplot(x="Fraud", y="Region", data =  claim1)

sns.lmplot(data= claim1, x='Call_details', y='Fraud')

plt.legend(fontsize=14)

# Density Plot and Histogram of all arrival delays
sns.distplot(claim1['Claim_Value'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4}) 

sns.barplot(x='Product_type',y='Fraud',data=claim1)

claim1['Claim_Value'].unique()
claim1['Fraud'].unique()
claim1['Product_type'].unique()
claim1.isnull().sum()

#converting into binary
lb=LabelEncoder()
claim1["Region"]=lb.fit_transform(claim1["Region"])
claim1["State"]=lb.fit_transform(claim1["State"])
claim1["Area"]=lb.fit_transform(claim1["Area"])
claim1["City"]=lb.fit_transform(claim1["City"])
claim1["Consumer_profile"]=lb.fit_transform(claim1["Consumer_profile"])
claim1["Product_category"]=lb.fit_transform(claim1["Product_category"])
claim1["Product_type"]=lb.fit_transform(claim1["Product_type"])
claim1["Purchased_from"]=lb.fit_transform(claim1["Purchased_from"])
claim1["Purpose"]=lb.fit_transform(claim1["Purpose"])

########## Median Imputation ############

claim1.fillna(claim1.median(), inplace=True)
claim1.isna().sum()

################## Standardization ##############

def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

df_norm = norm_func(claim1.iloc[:,:19])
df_norm.describe()

### New Dataset

claim2= pd.concat([df_norm, claim1.iloc[:,19]], axis = 1)



#########################################
############## Balancing ############## : oversampling performing better than smote and undersampling
#########################################


# Separate input features and target
x = claim2.iloc[:,:19]
y = claim2.iloc[:,19]

# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.40)

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

######### Create New Dataframe 
r2=result.drop(["index"],axis=1)

# check new class counts
r2.Fraud.value_counts()
#for chaking accuracy build model

# trying logistic regression again with the balanced dataset
y_train = r2.Fraud
X_train = r2.drop('Fraud', axis=1)

upsampled2= LogisticRegression(solver='liblinear').fit(X_train, y_train)

upsampled_pred = upsampled2.predict(X_test)

# Checking accuracy
accuracy_score(y_test, upsampled_pred)    
# f1 score
f1_score(y_test, upsampled_pred)
recall_score(y_test, upsampled_pred)

########
####SMOTE Balancing

# Separate input features and target
y = claim2.Fraud
X = claim2.drop('Fraud', axis=1)

# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=27)

sm = SMOTE(random_state=27, ratio=1.0)
X_train, y_train = sm.fit_sample(X_train, y_train)

#smote balancing
smote = LogisticRegression(solver='liblinear').fit(X_train, y_train)

smote_pred = smote.predict(X_test)

# Checking accuracy
accuracy_score(y_test, smote_pred)
    
# f1 score
f1_score(y_test, smote_pred)
   
recall_score(y_test, smote_pred)


###################################################################
########################## Feature Selection ######################
###################################################################


#Feature Selection using Tree Classifier
a = r2.iloc[:,0:19]  #independent columns
b = r2.iloc[:,-1]    #target column

model = ExtraTreeClassifier()
model.fit(a,b)

print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=a.columns)
feat_importances.nlargest(19).plot(kind='barh')


###############################################################
####################### Cross Validation ######################
###############################################################


colnames = list(r2.columns)
predictors = colnames[:19]
target = colnames[19]

Xx = r2[predictors]
Yy = r2[target]

#Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=300, random_state=24)
all_accuracies = cross_val_score(classifier,Xx,Yy,cv=10)
print(all_accuracies)
print(all_accuracies.mean()) #96.13

#Catboost
modell = ctb.CatBoostClassifier()
all_accuraciess = cross_val_score(modell,Xx,Yy,cv=10)
print(all_accuraciess)
print(all_accuraciess.mean())#95.39


#Adaboost Classifier
abc = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1)
all_accuraciesss = cross_val_score(abc,Xx,Yy,cv=10)
print(all_accuraciesss)
print(all_accuraciesss.mean())#93.61

#XGBClassifier
model = XGBClassifier()
all_accuracy = cross_val_score(model,Xx,Yy,cv=10)
print(all_accuracy)
print(all_accuracy.mean()) #94.11

#MLP clssifier
model4=MLPClassifier()
all_accuracyyy=cross_val_score(model4,Xx,Yy,cv=10)
print(all_accuracyyy)
print(all_accuracyyy.mean()) #94.88

#Ridge Classifier
model8=RidgeClassifier()
all_accuracyi=cross_val_score(model8,Xx,Yy,cv=10)
print(all_accuracyi)
print(all_accuracyi.mean()) #77.77%


#Random forest giving highest accuracy so we will build final model on Random Forest.



#############################################################################
############################## Final Model  Bulding #########################
#############################################################################


#Random Forest model building with feature selection
r2=r2.drop(["Product_category"],axis=1) #according to plot of feature selection change the column name
colnames = list(r2.columns)
predictors = colnames[:18]
target = colnames[18]

X = r2[predictors]
Y = r2[target]

#Train Test split
train,test = train_test_split(r2,test_size = 0.4,stratify=r2.Fraud)

###### Model building ######
rf = RandomForestClassifier(n_jobs=4,oob_score=True,n_estimators=200,criterion="entropy")

rf.fit(train[predictors],train[target])

rf.estimators_ # 
rf.classes_ # class labels (output)
rf.n_classes_ # Number of levels in class labels 
rf.n_features_  
rf.n_outputs_ # Number of outputs when fit performed
rf.oob_score_  #94.87
rf.predict(X)


pred_train=rf.predict(train[predictors])
pred_test = rf.predict(test[predictors])
pd.Series(pred_test).value_counts()
pd.crosstab(test[target],pred_test)
pd.crosstab(train[target],pred_train)

#f1 Score 
f1_score(test[target], pred_test)#96.70

#recall
recall_score(test[target], pred_test, average='weighted') #96.51

#precision
precision_score(test[target], pred_test, average='weighted')#96.88

# Accuracy = train
np.mean(train.Fraud == rf.predict(train[predictors]))#97.84

# Accuracy = Test
np.mean(pred_test==test.Fraud) # 96.79

#Good model


############################ **************** ##################


'''
trn=train.to_csv(r'E:\ADM\Excelr solutions\Warranty  claims project\tain.csv')
tst=test.to_csv(r'E:\ADM\Excelr solutions\Warranty  claims project\test.csv')

# =============================================================================
# =============================================================================
# #
# =============================================================================
# =============================================================================


