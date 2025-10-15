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
'''
Correlation Analysis: Use Pearson or Spearman correlation to identify relationships between your independent variables (temperature, speed, concentration) and the dependent variable (printability). A heatmap can visualize this.
The Pearson correlation coefficient measures the strength and direction of the linear relationship between two continuous variables. The value of the Pearson correlation (r) ranges from -1 to 1.
Scatter Plot: A quick visual inspection through scatter plots can help identify if a relationship between variables is linear or non-linear.
Spearman's Rank Correlation: This is a non-parametric measure that assesses the strength and direction of the monotonic relationship between two variables (i.e., whether they move together in the same or opposite direction, but not necessarily linearly).
Kendall’s Tau: Another non-parametric test that measures the strength of association between two variables, commonly used when the data contain outliers or ties.
 
Multivariate Analysis: Pair Plots: To see interactions between multiple variables.

'''
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






#==================================
#==================================
'''        More advanced        '''
#==================================
#==================================

'''
Data understanding and cleaning
 
Data Cleaning: Handle missing values, outliers, and anomalies. 
Make sure your data is correctly formatted and consistent.
 
Descriptive Statistics: Use mean, median, standard deviation, 
skewness, and kurtosis to get an overview of your data distribution.
 
Feature Distributions: Use histograms, density plots, or boxplots 
to understand the distribution of each variable.
 
Outlier Detection: Use methods like the Z-score, IQR method, or 
visualizations like box plots to identify and handle outliers.
 
Normalization or Standardization: Ensure your features (e.g., 
temperature, speed, concentration) are on a similar scale if required (especially important for ML).
 
 '''


'''
When output is continious our problem is regression-based , when our output is discrete
we said classification-based. But what about Inputs? input can be  countinious
or discrete so we can have multiple things
so this is nto always we have two arrays of floating numbers, but maybe we
have continious and discrete (we csalled them categorial variables).
Example --> adult incomes in united stated --> it said that workclass (state-gov) , education
can be bachelor .. , gender --> Male , fmaile and ...

so consider we want to have logistic regression which is
Y^ = w[0]*x[0] + w[1]*x[1] + .... + b 
so when teh all x are numebrs it is ok, but what abotu when one of
these x is mastwes or bachelors --> so e need to represnt our data in some
different way befor applying machine learning.

'''

#====================
'''
One-Hot-Encoding (Dummy Variables)

The Most common way to represent categorial variables is using one-hot-encoding
or one-out-of-N encoding.
This means that instead of one input column , we can have multipel columns
whgich has value of zero or one. now these values 0,1 make sense
iun fomula for lienar binary classification.

There are two ways , convert your data to a one-hot endocing of categorial variables
using pandas or Sk learn.
'''

import pandas as pd
# The file has no headers naming the columns, so we pass header=None
# and provide the column names explicitly in "names"
data = pd.read_csv("/home/andy/datasets/adult.data", header=None, index_col=False,
names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
'marital-status', 'occupation', 'relationship', 'race', 'gender',
'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
'income'])
# For illustration purposes, we only select some of the columns
data = data[['age', 'workclass', 'education', 'gender', 'hours-per-week',
'occupation', 'income']]
# IPython.display allows


#first check that check if a column actually contain meaningful categiorial data
#Liek this
print(data.gender.value_counts())

#so it means sometimes mayeb man , male or anything
data_dummies = pd.get_dummies(data)
#or specific
df = pd.get_dummies(df, columns=['city'], drop_first=True)

#** Note: always test and train also must be in same format

#----also we have this
df['gender'] = df['gender'].map({'male': 0, 'female': 1})



#for which you can specify which variables are continuous and which are discrete,
#or convert numeric columns in the DataFrame to strings.
# create a DataFrame with an integer feature and a categorical string feature
demo_df = pd.DataFrame({'Integer Feature': [0, 1, 2, 1],
'Categorical Feature': ['socks', 'fox', 'socks', 'box']})
display(demo_df)




#so if your data even have that 0 ,1 or anything and integer instead of string
#you mjst specify that
demo_df['Integer Feature'] = demo_df['Integer Feature'].astype(str)
pd.get_dummies(demo_df, columns=['Integer Feature', 'Categorical Feature'])




#----so for categorial columns
#1.1 Nominal (Unordered) Categorical Features
#One-Hot Encoding (Dummy Encoding)
pd.get_dummies(df['Color'], drop_first=False)

#so if our data has a lot of options
#1.2.Binary Encoding (for high-cardinality nominal features)
from category_encoders import BinaryEncoder
encoder = BinaryEncoder(cols=['Country'])
df_encoded = encoder.fit_transform(df)

#more compact than oNE-HOT ENCODING

#2-Ordinal (Ordered) Categorical Features
'''
Ordinal variables represent categories with a meaningful order or 
ranking, but not necessarily uniform spacing.
Education → {High School < Bachelor < Master < PhD}
Size → {Small < Medium < Large}
'''
#Ordinal Encoding (Label Replacement Based on Order)
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder(categories=[['High School', 'Bachelor', 'Master', 'PhD']])
df['Education_encoded'] = encoder.fit_transform(df[['Education']])





#-------------------------
'''
Continious Inputs
'''
#--------------------------
#we talked about that very very , 
#like
#before that it must be cleaned (Not empty ,..)
#outlier and ... --> all correlation, statistics tests and then
#Feature Scaling --> Minmax and .....
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler = MinMaxScaler()
scaler = RobustScaler()

df[['age', 'income']] = scaler.fit_transform(df[['age', 'income']])




#--interaction with polynomial
#Polynomial Features
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
df_poly = poly.fit_transform(df[['feature1', 'feature2']])


#sometimes for reducing skweness
#Log Transformation
import numpy as np

df['income_log'] = np.log1p(df['income'])  # log(1 + x) to handle zero values


#----
#one way to make linear models more powerful on continious data is 'bining'
bins = np.linspace(-3, 3, 11)
print("bins: {}".format(bins))
#bins: [-3. -2.4 -1.8 -1.2 -0.6 0. 0.6 1.2 1.8 2.4 3. ]

which_bin = np.digitize(X, bins=bins)

which_bin = np.digitize(X, bins=bins)
print("\nData points:\n", X[:5])
print("\nBin membership for data points:\n", which_bin[:5])

'''
Data points:
[[-0.753]
[ 2.704]
[ 1.392]
[ 0.592]
[-2.064]]
Bin membership for data points:
[[ 4]
[10]
[ 8]
[ 6]
[ 2]]
What we did here
'''
#now it is liek categorial so
from sklearn.preprocessing import OneHotEncoder
# transform using the OneHotEncoder
encoder = OneHotEncoder(sparse=False)
# encoder.fit finds the unique values that appear in which_bin
encoder.fit(which_bin)
# transform creates the one-hot encoding
X_binned = encoder.transform(which_bin)

#now it is 0 1 0 1 0 1 0 10 


'''
the linear regression model and the decision tree make exactly the same 
predictions. For each bin, they predict a constant value. As features 
are constant within each bin, any model must predict the same value for 
all points within a bin. Comparing what themodels learned before binning 
the features and after, we see that the linear model became much more 
flexible, because it now has a different value for each bin, while the 
decision tree model got much less flexible. Binning features generally  has
no beneficial effect for tree-based models, as these models can learn to 
split up the data anywhere

'''

#first scale and then polynomial [in book]



#Univariate Nonlinear Transformations
#the polynomial is onlhy squared, cubed features 
#but nonlinear means liek log, exp , or sin and ...



#==============================================
#------------Automatic Feature Selection-------
'''
With so many ways to create new features, you might get tempted to 
increase the dimensionality of the data way beyond the number of 
original features.


However, adding more features makes all models more complex, and so 
increases the chance of overfitting.


When adding new features, or with high-dimensional datasets in general,
it can be a good idea to reduce the number of features to only the most useful ones,
and discard the rest. This can lead to simpler models that generalize better



OVER ==> untill now you want to add features or doing anythign but 
sometimes some of these adding (or maybe in original data) some
columsn nto only no extra information but also confuse ythe model
so we must has feature selection methods and we have 3 method
1-univariate statistics
2-model-based selection
3-iterative selection1

'''


'''
1-Univariate Statistics
'''







'''
2-Model-Based Feature Selection
'''







'''
3-Iterative Feature Selection
'''





#======================
'''
Utilizing (Incorporation) Expert Knowledge
'''







