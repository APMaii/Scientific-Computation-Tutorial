'''
it msut fill with all necessary data pre processing before ML (after cleaning data with pandas
'''

#--------Reading Data from pandas
import pandas as pd
filepath='something'
data=pd.read_csb(filepath)
print(data.iloc[:5])

data=pd.read_csv(filepath,sep='\t')

#space speerated
data=pd.read_csv(filepath,dalim_whitespace=True)

#dont use first row for column namee
data=pd.read_csv(filepath,header=None)

#specify the column anme
data=pd.read_csv(filepath,names=['Name1','Name2'])

#custom missing value
data=pd.read_csv(filepath,na_values=['NA',99])

#read JSON
data=pd.read_json(filepath)

#or from DATABASE LIKE
import sqlite3 as sq3
#initialize path to sqlite databse
path='data/classic_rock.db'
#creat conection to SQL database
con=sq3.Connection(path)
query=''' SELECT * FROM rock_songs;
'''
data=pd.read_sql(query,con)


#or API
#it is liek click on the download
data_url= 'https:// ..... '
df=pd.read_csv(data_url,header=None)





#---see head and tail
housing.head(5)
housing.tail(5)

housing.info() #--> non-null, type
housing.describe() #--> means and ...
#or we can get for specific column
housing["SalePrice"].describe()
#for categorial
housing["Sale Condition"].value_counts()



#-----CLEANING------
#we talk about in pandas but again review-------

#---------EMPTY CELLL
data.info()



housing.isnull() #true false
housing.isnull().sum() # for eahc solum write that
housing.isnull().sum().sort_values(ascending=False) # from top to down to see the most





data.dropna(inplace=True)
data.info()
data.fillna(10,inplace=True)


mymean=data['Temp'].mean()
new_data=data.fillna(mymean)
new_data=data.fillna(method='ffill') #haraj khalie gjhablairo mizare
new_data=data.fillna(method='bfill')

#also for one specific column
df['age'].fillna(df['age'].mean(), inplace=True)
# For categorical columns
df['gender'].fillna(df['gender'].mode()[0], inplace=True)

#or
housing.dropna(subset=["Lot Frontage"])



#-------Wrong format
data['temp']=pd.to_numeric(data['temp'])

df['age'] = df['age'].astype(int)  # Convert to integer
df['date'] = pd.to_datetime(df['date'])  # Convert to datetime



#----Logical-----
#access
df.iloc['charles'] 
df.iloc[1:3]

#filter
df[df["birthyear"] < 1990]
df[df['price'] > 100]
#--for multiple
df[(df['price'] > 100) & (df['stock'] > 0)]


#new column
people["age"] = 2018 - people["birthyear"]  # adds a new column "age"
df['category'] = df['age'].apply(lambda x: 'Senior' if x > 30 else 'Young')
df.loc[df['age'] > 30, 'status'] = 'experienced'
df.loc[df['price'] > 100, 'discount'] = 0

birthyears = people.pop("birthyear")
data.drop(columns='Modulus',inplace=True)
#inpalce=tru yani agha taghirati k goftmno roohamini k dot xadam (data ) emal kon man zarfe jhadid nmikham
zarf=data.drop(index=1)


def adjust_price(row):
    if row['category'] == 'Clearance' and row['price'] > 100:
        return row['price'] * 0.8
    return row['price']

df['price'] = df.apply(adjust_price, axis=1)


#hazf
for x in data.index:
    if data.loc[x,'temp']<0:
        data.drop(x,inplace=True)

#jaygozin koni
for x in data.index:
    if data.loc[x,'temp']<0:
        data.loc[x,'temp']=20 #y amiangine ...


#-----du-plicatd
duplicate = housing[housing.duplicated(['PID'])]
dup_removed = housing.drop_duplicates()
removed_sub = housing.drop_duplicates(subset=['Order'])



#------FINALLY-----
data.reset_index(drop=True,inplace=True)



#-------For visulaization we have a lot of this meythods
import seaborn as sns
import matplotlib.pyplot as plt
sns.histplot(df['price'])
sns.boxplot(x='category', y='price', data=df)
sns.pairplot(df)





#-----Outlier handling----------------
#----plot approach
data=housing['PID']
import searborn as sns
sns.displot(data,bins=20)
sns.boxplot(data)


#-----IQR
import numpy as np
q25,q50,q75=np.percentile(data,[25,50,75])
iqr=q75-q25
minn=q25-1.5*(iqr)
maxx=q75-1.5*(iqr)

print(minn,q25,q50,q75,maxx)
[x for x in data['Unpleymoent'] if x>max]





#--but for statistics best one is
sp_untransformed = sns.distplot(housing['SalePrice'])

#we can use log to see that it is ok or not
log_transformed = np.log(housing['SalePrice'])
sp_transformed = sns.distplot(log_transformed)
print("Skewness: %f" % (log_transformed).skew())


print("Skewness: %f" % housing['SalePrice'].skew())


#-----correlation-----

hous_num_corr = hous_num.corr()
#or
plt.figure(figsize=(18,18))
sns.heatmap(data.corr(),annot=True,cmap='RdYlGn')

plt.show()


#-----group by
geo = data.groupby('GEO')




#------------------FEATURE ENGINEERING----------------------------
'''
clean and eda on data. before mdoeling-->aw material
we must feature enginering and variable transformation
feature encoding ( must be numerical or categorial)
feature scaling --> same scal enot or everything


models used in ML-->make soem assumption

a common example is the Linear regression model:
assume a lienar relatinship between observation and target (outcome)

we have lienar mdeodel feature(x2,x2) to label(y)
t(x)=b0 + b1x1 +b2x2

b=(b0,b1,b2)-->models parmaters that is traind and learned.

we can transform x1 and x2 and but stil we have a lenar elationship

'''

#-----data distribution
from numpy import log,loglp 
#loglp () if we have 0 in dtaaset
from scipy.stats import boxcox #finding some to get skwed to normal)
#---log
#for instance positive skwed
sns.distplot(data,bins=20)
import math
log_data=[math.log(d) for d in data['Unemplyment']]
sns.displot(log_data,bins=20)


#or

from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=2) #this is fit function

poly=poly.fit(x_data)
x_poly=poly.transform(x_data)



#----feature scaling
norm_data = MinMaxScaler().fit_transform(hous_num)
scaled_data = StandardScaler().fit_transform(hous_num)




#encoding-->often applied to categorial feature

#5-----encoding------
#Option A: Label Encoding
df['gender'] = df['gender'].map({'male': 0, 'female': 1})
#one hot encoding
df = pd.get_dummies(df, columns=['city'], drop_first=True)

'''

Feature scaling
for continous :
    with distplot see distribution
    using np.log or other 
    then minmax scaler and standard scaler or robust
    then we can  interation or polynominal (for liinear mdoel taht assume
                                            there is linear relationship
                                            betwen them)

for categorial-->featur eencoding:
    nominal(without order)-->binary or hot necoding (get_dummy)
    ordinal->ordinal ordered (replace)
'''




#==================================
#==================================
'''        More advanced        '''
#==================================
#==================================














