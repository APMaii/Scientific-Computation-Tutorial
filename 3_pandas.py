#from Book and our tutorials
#also from book ML
#AND ALSO
https://github.com/ageron/handson-ml2/blob/master/tools_pandas.ipynb
'''
Before Anything you must clean your data with Pandas
What is Pandas?

Actually something like numpy but not only the index for rows and index for columns
instead of index, we need name for columns.
'''

#---------importing pandas-----------

import pandas as pd


#-----Series objects--------
#A Series object is 1D array, similar to a column in a spreadsheet (with a column name and row labels).
s = pd.Series([2,-1,3,5])
'''
0    2
1   -1
2    3
3    5
dtype: int64
'''


#you can cast to numpy  like 1D nparray
np.exp(s)

#you can also do something to them
s + [1000,2000,3000,4000]
'''
0    1002
1    1999
2    3003
3    4005
dtype: int64
'''


s + 1000
'''
0    1002
1     999
2    1003
3    1005
dtype: int64
'''

s < 0
'''
0    False
1     True
2    False
3    False
dtype: bool
'''




a=pd.Series([10,20,30,40,50,60])
a+2
a*100
a/10
a**2
a//10


#comparison
'''
0    True
1    True
2    True
3    True
4    True
5    True
dtype: bool
'''




a=[10,20,30,40,50,60]

s=pd.Series(a)


import numpy as np

a=np.array([10,20,30,40,50,60])
s=pd.Series(a)



weights = {"alice": 68, "bob": 83, "colin": 86, "darwin": 68}
s3 = pd.Series(weights)



meaning = pd.Series(42, ["life", "universe", "everything"])
'''
life          42
universe      42
everything    42
dtype: int64
'''




#Index labels==================================
#you can label the indexes like dictionary
s2 = pd.Series([68, 83, 112, 68], index=["alice", "bob", "charles", "darwin"])
#*******
#also index, dtype it has for them
#also it can has name
#A Series can have a name:s6 = pd.Series([83, 68], index=["bob", "alice"], name="weights")



#you can simply call the index or name
s2[0] #68
s2['alice'] #68



#------LOC and Iloc
#iloc --> for index
#loc ---> for only name
s2.loc['alice']
s2.loc[0] #error

s2.iloc[0] #68
s2.iloc['alice'] #error

s2.iloc[1:3]
'''
bob         83
charles    112
dtype: int64
'''



print(s2.keys())
#Index(['alice', 'bob', 'charles', 'darwin'], dtype='object')





#-------------------------Time range------------------------------------
dates = pd.date_range('2016/10/29 5:30pm', periods=12, freq='H')
'''
DatetimeIndex(['2016-10-29 17:30:00', '2016-10-29 18:30:00',
               '2016-10-29 19:30:00', '2016-10-29 20:30:00',
               '2016-10-29 21:30:00', '2016-10-29 22:30:00',
               '2016-10-29 23:30:00', '2016-10-30 00:30:00',
               '2016-10-30 01:30:00', '2016-10-30 02:30:00',
               '2016-10-30 03:30:00', '2016-10-30 04:30:00'],
              dtype='datetime64[ns]', freq='H')

'''
temperatures = [4.4,5.1,6.1,6.2,6.1,6.1,5.7,5.2,4.7,4.1,3.9,3.5]

#also you can create this with each other
#it means that the first is valeus and the dates is the index

temp_series = pd.Series(temperatures, dates)
temp_series
'''
2016-10-29 17:30:00    4.4
2016-10-29 18:30:00    5.1
2016-10-29 19:30:00    6.1
2016-10-29 20:30:00    6.2
2016-10-29 21:30:00    6.1
2016-10-29 22:30:00    6.1
2016-10-29 23:30:00    5.7
2016-10-30 00:30:00    5.2
2016-10-30 01:30:00    4.7
2016-10-30 02:30:00    4.1
2016-10-30 03:30:00    3.9
2016-10-30 04:30:00    3.5
Freq: H, dtype: float64
'''


#-----It has a lot of functions the series itself--------
s2.abs()
s2.add()
s2.div()
s2.divide() #similar
s2.divmod() #integer
s2.multiply() #*
s2.mul()
s2.pow()

s2.pop() #remove
s2.clip()  #thresholding


s2.all()
s2.any()
s2.max()
s2.min()
s2.argmax()
s2.argmin()
s2.astype() #dtype
s2.view()
s2.copy()
s2.keys()
s2.items() #for i,j in 
s2.apply()
s2.filter()
s2.isin()
s2.isna()
s2.isnull()
s2.fillna()
s2.drop()
s2.drop_duplicates()
s2.dropna()
s2.ffill()
s2.bfill()



#==============================
#==============================
#==============================
#======DataFrame objects=======
#==============================
#==============================
#==============================
'''
A DataFrame object represents a spreadsheet, with cell values,
column names and row index labels. You can define expressions 
to compute columns based on other columns, create pivot-tables,
group rows, draw graphs, etc. You can see DataFrames as dictionaries of Series.

'''
#now it is somethign like 2D numpy array that you can have name of rows (index) and name of columns

#here 
#Imagine you have the dictionary that 
people_dict = {
    "weight": pd.Series([68, 83, 112], index=["alice", "bob", "charles"]),
    "birthyear": pd.Series([1984, 1985, 1992], index=["bob", "alice", "charles"], name="year"),
    "children": pd.Series([0, 3], index=["charles", "bob"]),
    "hobby": pd.Series(["Biking", "Dancing"], index=["alice", "bob"]),
}


#here you can say
people_dict['weight'] #and it get you the 68, 83,112 that has 3 index

#but the type is dictionary



#so now you can convert to DatFrame
people = pd.DataFrame(people_dict)
people

#nwo the same


#accessing the columns
people["birthyear"]

#both columns
people[["birthyear", "hobby"]]



#----------
#also you can create with numpy arrays
#****
values = [
            [1985, np.nan, "Biking",   68],
            [1984, 3,      "Dancing",  83],
            [1992, 0,      np.nan,    112]
         ]


d3 = pd.DataFrame(
        values,
        columns=["birthyear", "children", "hobby", "weight"],
        index=["alice", "bob", "charles"]
     )





#-----ADVNACED-----
#-----Multi-indexing

d5 = pd.DataFrame(
  {
    ("public", "birthyear"):
        {("Paris","alice"):1985, ("Paris","bob"): 1984, ("London","charles"): 1992},
    ("public", "hobby"):
        {("Paris","alice"):"Biking", ("Paris","bob"): "Dancing"},
    ("private", "weight"):
        {("Paris","alice"):68, ("Paris","bob"): 83, ("London","charles"): 112},
    ("private", "children"):
        {("Paris", "alice"):np.nan, ("Paris","bob"): 3, ("London","charles"): 0}
  }
)
d5


#-----again
values = [
            [1985, np.nan, "Biking",   68],
            [1984, 3,      "Dancing",  83],
            [1992, 0,      np.nan,    112]
         ]

df = pd.DataFrame(
        values,
        columns=["birthyear", "children", "hobby", "weight"],
        index=["alice", "bob", "charles"]
     )


#----Accessing the columns---------


df['birthyear'] #---> it get back teh all rows ( in the type of serie)

df[['weight','hobby']] #it get back both column -> as dataframe type



df.loc["charles"] #--> it get back teh rows (all columns)
df.loc[0] #error


df.iloc[0] # it get back teh rows (all columns) but i s index
df.iloc['charles'] #error
df.iloc[1:3]


#------filtering------------
#filter the all people that its columns has some feature
#also and , or 
df[df["birthyear"] < 1990]



#adding new column easy-----
people["age"] = 2018 - people["birthyear"]  # adds a new column "age"
people["over 30"] = people["age"] > 30      # adds another column "over 30"


df[df['age'] > 30]
df[(df['age'] > 25) & (df['name'] == 'Alice')]
df[df['age'].isin([25, 35])]
df[df['age'].between(26, 34)]


#data manupulation
df['age_plus_5'] = df['age'] + 5
df['category'] = df['age'].apply(lambda x: 'Senior' if x > 30 else 'Young')



#---or
# Modify a column conditionally
df['category'] = df['age'].apply(lambda x: 'Senior' if x > 30 else 'Young')

# Using .loc to conditionally fill or update values
df.loc[df['age'] > 30, 'status'] = 'experienced'
df.loc[df['age'] <= 30, 'status'] = 'junior'

# Fill NaNs in specific rows based on condition
df.loc[df['age'] > 25, 'bonus'] = df.loc[df['age'] > 25, 'bonus'].fillna(0)



#remove----------
birthyears = people.pop("birthyear")
del people["children"]


#_----REMOVE------
a=np.random.uniform(0,10,size=(50,3))
data=pd.DataFrame(a,columns=['Temp','Time','Modulus'])

data2=data.drop(columns='Modulus')
#return mide yani bayad brizi otoye ye zarfe dg
#datae taghir nrkde ama data2 hazf shod

#hamishe aksare tabe haye pandas replace=False
#age nakhay dobare too zarfe jadid brizi va hamon
#khode datat taghir kone -->true

data.drop(columns='Modulus',inplace=True)
#inpalce=tru yani agha taghirati k goftmno roohamini k dot xadam (data ) emal kon man zarfe jhadid nmikham

zarf=data.drop(index=1) #radif ro hzzf krd
data.drop(index=1,inplace=True)


#yadet nare......
data.reset_index(drop=True,inplace=True)






#by default when you add column it is like append , so you must use this for more order
df.insert(1, "height", [172, 181, 185])



#---also you can use assign to not apply just return new df
df.assign(
    body_mass_index = people["weight"] / (people["height"] / 100) ** 2,
    has_pets = people["pets"] > 0
)


#---evaluating a expression
df.eval("weight / (height/100) ** 2 > 25")
'''
alice      False
bob         True
charles     True
dtype: bool

'''


#---it create new column--
df.eval("body_mass_index = weight / (height/100) ** 2", inplace=True)



#query() let you filter and get you back
df.query("age > 30 and pets == 0")



#---sorting----
df.sort_index(ascending=False)


#also you can sort the columns , instead of rows --> axis=1
df.sort_index(axis=1)

#----*******
#all the functions return rew dataframe but if you wnat you can click on the inplace=True
df.sort_index(axis=1, inplace=True)


#or you can sort only by one column
df.sort_values(by="age", inplace=True)


df.sort_values('age', ascending=False)
df['rank'] = df['age'].rank(ascending=False)




#------Instead of using Plot and other things
#----you can use nthe functions inside them

#================================
#================================
#-----Plotting a DataFrame------
#================================
#================================
people.plot(kind = "line", x = "body_mass_index", y = ["height", "weight"])
plt.show()


people.plot(kind = "scatter", x = "height", y = "weight", s=[40, 120, 200])
plt.show()






#----Operations on DataFrames---------
grades_array = np.array([[8,8,9],[10,9,9],[4, 8, 2], [9, 10, 10]])
grades = pd.DataFrame(grades_array, columns=["sep", "oct", "nov"], index=["alice","bob","charles","darwin"])



#You can apply NumPy mathematical functions on a DataFrame: the function is applied to all values:------
np.sqrt(grades)


grades + 1
grades >= 5

grades.mean()
(grades > 5).all()
(grades > 5).all(axis = 1)
(grades == 10).any(axis = 1)

grades - grades.mean() 







#========================================
#========================================
#--------------Data Cleaning-------------
#========================================
#========================================
jadval=pd.DataFrame([[10,20,30,40],[50,60,70,80]],index=['a','b'],columns=['dama','feshar','time','output'])
#ham indexash (radifahs) esm dre ham column ha soton ha esm dre


#or
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')


#------accesss--- dastresssiiii
#az numpyu  yadete vghty do bodi shodi 
#hamrogeh gfotn acces bgi kdoom radif kdoom sotoon



#--------row------ radif dasteresi peyda koni
#b rdovomin radif
#alan msihe radife b ya b soorat eghaidmi bgam radife 1
jadval.loc['b'] #50,60,70,80
jadval.iloc[1] # 50,60,70,80 radife 1 mikham kolesho

#.loc --> ba esmesh
#ag bkhay ba index seda koni ---> .iloc


#----------column ---->yek sotoon
jadval['dama']
jadval['feshar']
#doat sotoon
jadval[['dama','feshar']]

#pad masalan kahte 161 mige agha sootone feshar ro az jadval koelsho bde

#ya yek element --->
#-----element--- yani yk adad bkhay bkshi biron
#msln 50 ro mikham
jadval.loc['b','dama'] #Out[67]: 50
#ag bkhay indexi bnhsh dastresi peyda koni?
jadval.iloc[1,0] #Out[68]: 50

jadval.loc[ : ,'dama'] #mese 160 fght yek jadval


'''
.doc .docx .txt   -->matne tooshoi dar bairi
zarf=open('directory/ / / /khode_file.format')
zarfe str--->


.csv .xlsx
#jadvale , str  , dataframe 
a=pd.read_csv('directory/ / / /khode_file.csv')
b=pd.read_excel('directory/ / / /khode_file.xlsx')


.png .jpg .mp4
open()

ketabkhaneye opencv -->koli pardazeshe tasvir rule based
AI (ML BASED)
'''

pd.read_csv('directory')

data.max(axis=0)  #too har soton max ro mide
data.max(axis=1) #too har radi max ro mide

#hamchnin
data['dama'].max() #Out[82]: 90


import pandas as pd
# Sample DataFrame
df = pd.DataFrame({
    'department': ['Sales', 'Sales', 'HR', 'HR', 'IT', 'IT'],
    'employee': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank'],
    'salary': [50000, 55000, 48000, 52000, 60000, 62000]
})

# Group by department and get average salary
df.groupby('department')['salary'].mean()
'''
.mean()	Average
.sum()	Total sum
.count()	Number of entries
.min()	Minimum value
.max()	Maximum value
.median()	Median
.std()	Standard deviation
'''
# Total and average salary per department
df.groupby('department').agg({
    'salary': ['sum', 'mean', 'max']
})


print(df.info())
print(df.describe())
print(df.isnull().sum())  # check missing values
print(df.duplicated().sum())  # check for duplicates




print(df.describe())

#to include
df.describe(include='all')


df['price'].describe()


'''
Mean: df['price'].mean()
Median: df['price'].median()
Mode: df['price'].mode()
Standard Deviation: df['price'].std()
Variance: df['price'].var()
Min / Max: df['price'].min() / df['price'].max()
df['price'].skew()      # Skewness
df['price'].kurtosis()  # Kurtosis
df['price'].quantile(0.25)  # 25th percentile
df['price'].quantile([0.25, 0.5, 0.75])  # Q1, Q2 (median), Q3

Correlation Matrix: df.corr()


'''


df.corr()       # Pearson correlation between numeric columns
#Use df.corr(method='spearman') or 'kendall' for other correlation types.

df.cov()        # Covariance matrix


#custom statistics
df[['price', 'quantity']].apply(lambda x: x.max() - x.min())  # Range per column



df['category'].unique()          # List of unique values
df['category'].nunique()         # Count of unique values
df['category'].value_counts()    # Frequency of each value



(df['price'] > 100).sum()        # Count of entries where price > 100



df.groupby('category')['price'].mean()
df.groupby('category').agg({
    'price': ['mean', 'min', 'max'],
    'quantity': 'sum'
})


df.isnull().sum()          # Count of missing values per column
df.isnull().mean()         # Proportion of missing values


#_---visual----


import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df['price'])
sns.boxplot(x='category', y='price', data=df)
sns.pairplot(df)







#yani toye data , sotone dama maxz kodome
#bishtar miran too ye sotoon va max ro mikhan

#kole datat ro sort koni--------
#tartib bedi
data=pd.DataFrame([[10,20,30,100],[50,60,70,40],[90,100,110,30]],index=['a','b','c'],columns=['dama','feshar','time','output'])

new_data=data.sort_values(by='output')


#concat dota darfram join

df1=pd.DataFrame([1,2])
df2=pd.DataFrame([3,4])

zarf=pd.concat([df1,df2])



#_----REMOVE------
a=np.random.uniform(0,10,size=(50,3))
data=pd.DataFrame(a,columns=['Temp','Time','Modulus'])

data2=data.drop(columns='Modulus')
#return mide yani bayad brizi otoye ye zarfe dg
#datae taghir nrkde ama data2 hazf shod

#hamishe aksare tabe haye pandas replace=False
#age nakhay dobare too zarfe jadid brizi va hamon
#khode datat taghir kone -->true

data.drop(columns='Modulus',inplace=True)
#inpalce=tru yani agha taghirati k goftmno roohamini k dot xadam (data ) emal kon man zarfe jhadid nmikham

zarf=data.drop(index=1) #radif ro hzzf krd
data.drop(index=1,inplace=True)


#yadet nare......
data.reset_index(drop=True,inplace=True)







#moshekle data chia mione bashe????
'''
1-empty cell   yek adad khali bashe (khataye ensani, khataye dastgah, import) NAN None
2-wrong format    #asdad bashe str has
3-wrong data  (dama ha hame balaye 0 , -10)
4-duplicated (tekrari)


dalilesh harchi mikahd bashe
ama in 4 ta mroed -->moshekel data
pas---> ag ina residgegi nashan
momekne mdoele ma k data ro migire asan run nashe, moshekl dahste bashe , accuracy paen bashe va va va....

'''



#------1-EMPTY CELL----
a=np.random.uniform(0,10,size=(50,3))
data=pd.DataFrame(a,columns=['Temp','Time','Modulus'])
data.loc[5,['Temp']]=None
data.loc[17,['Temp']]=None
data.loc[20,['Temp']]=None




data.info()

#1.1.tashkhis 
#felan ba data.info()
#empty cell haro tshkhis dadi

#sade tarin akri k mitoni koni
#bgei agha boro harjaaa harjaa empty cell has rmemoev kon oon radifo
data.dropna(inplace=True)
data.info()


#sadettarin ine ye adad khdoet bzari
data.fillna(10,inplace=True)

#pishrafte tar
#hey zarf misazam shoam inpalce=True

mymean=data['Temp'].mean()

new_data=data.fillna(mymean)

new_data=data.fillna(method='ffill') #haraj khalie gjhablairo mizare

new_data=data.fillna(method='bfill')

new_data.info()





#or fill teh one specific columns
# For numerical columns
df['age'].fillna(df['age'].mean(), inplace=True)

# For categorical columns
df['gender'].fillna(df['gender'].mode()[0], inplace=True)




#2------wrong format gahlate

#temp--.float ,int  str 


data=pd.DataFrame([['1',2,3],['2',5,6]],columns=['temp','pressure','modulus'])

data.info()



data['temp']=pd.to_numeric(data['temp'])

df['age'] = df['age'].astype(int)  # Convert to integer
df['date'] = pd.to_datetime(df['date'])  # Convert to datetime







#-----3-moshekle mantehi dre
data=pd.DataFrame([[20,2,3],[50,5,6],
                   [30,2,3],[70,5,6],
                   [20,2,3],[80,5,6],
                   [90,2,3],[100,5,6],
                   [20,2,3],[24,5,6],
                   [-10,2,3],[28,5,6],
                   [22,2,3],[20,5,6],
                   [20,2,3],[20,5,6]],columns=['temp','pressure','modulus'])

#n az format na khalie na hcihi
#khode data msohekel m,anteghi
#hala b har dalili

#aval tashkhis

#for bzni bri too tmep ha va bbini
count=0
for x in data.index:
    if data.loc[x,'temp']<0:
        count=count+1

print(count) # 12 doone damaye zxire sefrf vodjod dare

import matplotlib.pyplot as plt
y=data['temp']
plt.plot(y,'o')


#hazf
for x in data.index:
    if data.loc[x,'temp']<0:
        data.drop(x,inplace=True)

#jaygozin koni
for x in data.index:
    if data.loc[x,'temp']<0:
        data.loc[x,'temp']=20 #y amiangine ...
        

#-------
df[df['price'] > 100]

#--for multiple
df[(df['price'] > 100) & (df['stock'] > 0)]



#_----
#where3 price is <100 , is 100
# Set discount to 0 where price is more than 100
df.loc[df['price'] > 100, 'price'] = 0

#or discount
df.loc[df['price'] > 100, 'discount'] = 0


#change two columns
df.loc[df['price'] > 100, ['discount', 'status']] = [0, 'high price']


# Fill NaN in 'discount' with 0 only where 'price' > 100
df.loc[(df['price'] > 100) & (df['discount'].isna()), 'discount'] = 0


#or said pric * 0.9
df.loc[df['category'] == 'Clearance', 'price'] *= 0.9


#or custome
def adjust_price(row):
    if row['category'] == 'Clearance' and row['price'] > 100:
        return row['price'] * 0.8
    return row['price']

df['price'] = df.apply(adjust_price, axis=1)










#4------Duplicated
data.drop_duplicates(inplace=True)

#--------BAD AZ DATA CLEANING-------
#vaghty akret tamom shod
data.reset_index(drop=True,inplace=True)




#5-----encoding------
#Option A: Label Encoding
df['gender'] = df['gender'].map({'male': 0, 'female': 1})


#one hot encoding
df = pd.get_dummies(df, columns=['city'], drop_first=True)







#finally
data.reset_index(drop=True,inplace=True)


print(df.info())
print(df.head())


#-----finally save
df.to_csv('cleaned_data.csv', index=False)
df.to_excel('report.xlsx', index=False)
df.to_json('data.json', orient='records')



'''
PANDAS ---> SERIES (NP.ARRAY 1 BODY K MITONE INDEX NAME BZRI)
DATAFRAME --> NP.ARRAY 2 BODY , COLUMN , INDEX ESM BZZARI

VA .. MITONIE EXCELETO, CSVITO BNA ESTEFADE AZ TABEYE READ_CSV , REWAD_EXCEDL
b soorate yek datafrasme too yek zarf brizi


vaghty rikhti 
ba info() mitoni info bgriii


#step 0 ghabla z ahrkari bayuad shoam  chika --.data cleaning
1-empty cell ---> info() , dropna() , fillna( mitoni adad bdi , mean, method='ffil','bbil')
2-wrong format --> to_numeric() rooye oon soton anjam bdi
3-worng data -->logical manteghi tashkhis--> for bzni (count) . plt.plot() / eghdam-->hazf koni drop() loc =
4-duplicated ->datat kam bodo duplicated() true false / ag ziad bod agar mikhasti k hazf koni . drop_duplicated()
 
va abd az tamiz krdne hameye ina
az reset_index yadet nare estefade koni


hala dateye to amade ye kar krdne

np , list series--> ketabkhone h aestefade koni



'''


#------------------
#==================
'''

request-->yek tabe benevisid k chanta application dahste bashe

a=pd.read_excel('')
dar a data fram dre


def 





'''



data=pd.read_excel('/Users/apm/Desktop/MAIN/Hojjat Emami/Span network/Compact zip file/open that/experimental/f5.xlsx')


#tdatash column stress , strain
#---harchi


'''

def(data,application):
    
    if application=='calculation':
        
        
    elif application=='plot':
        
    
    





'''

tabeyeshoma(data,'plot')
tabeyeshoma(data,'calcualtion')

'''
Tabe ee bayad besaziiid k dota vroodi bgire

yeki data
yeki oon kari k karbar donmableshe

'''

#hezaran data
#stress strain
#wave 
#ftri 
#

data=pd.read_excel('/Users/apm/Desktop/MAIN/Hojjat Emami/Span network/Compact zip file/open that/experimental/f5.xlsx')



def test(data,application):
    if application=='plot':
        x=data['Wavenumber  (cm -1)']
        y=data['Transmitance (a.u.)']
        plt.plot(x,y)
        plt.show()     
        
    elif application=='calcualt':
        print('salam')
    
    
    




data=pd.read_excel('/Users/apm/Desktop/MAIN/Hojjat Emami/Span network/Compact zip file/open that/experimental/f5.xlsx')
data.columns

test(data,'plot')

test(data,'calcualt')



'''

def test(data,application):
    
    if applciation=='plot':
        
    elif applciation=='max':
        yekare dg
        
    elif applciaation =='ssjdhn'
    
    
sakhataaaaaarrrrr


nesbat b rehste , alaghe , akri k mikahy anjam bdid
dorooze dg bbande paym bdid va bgid dar ch zmain eee kar mikonid

oon dastgahe mrotabety , dataye mortabete


polymer, stress strain 
dastgah 

data --?
data.csv

soton --> stresss 10,20,30,40,50,60,...
soton-->strain  1,2,3,4,5,6,7,8,9,......



#rasmesh konim 
rasmesh ch sheklie



#max stress

#max strain

#....



#-------B BESMELLA.....

data= dataframe yek sootonm stress yek soton strain

def Stress_Strain(data,application):
    stress=np.array(data['stress'])
    strain=np.array(data['strain'])
    
    if application=='plot':
        plt.plot(strain,stress)
        plt.title(-----)
        ....
        plt.show()
        
    elif application=='maxstress':
        maxstress=stress.max()
        return maxstress
    
    elif application=='maxstrain':
        maxstrain=strain.max()
        return maxstrain
    
    
    elif application=='alipilehvar':
        apm=maxstreeess+maxstrain
        return apm
    
    applicatyion---- 
    
    
dataee k entekhab krdiddd

data=pd.read_
Stress_Strain(data,'plot')
    

***naizi nsit too dle tabe dataframe baz 




#aval bgi ch dataee mikhay akr koni????


taraf ch dataee mitone b tabeye man bde


az koja entkehab


made nazare khdoet dare -->search bzn , ide bgir , ide rahnamaee
agar chzii monaeb peyda nkrdi


ai.2024.pilehvar@gmail.com 2 rooze


FTIR

shedate jaz
toole moji 

dota sotoon

'''


#entkehabe esme sootn

def FTIR(data,application):
    '''
    data= .csv .excell
    columns name=toolemoj , shedat
    application:
        plot --> drawng the wavelength on intensity
        maxintens--> maximum ....
        min....
        constant -->
        formula---->>
        
    
    '''
    x=np.array(data['toolemoj'])
    y=np.array(data['shedat'])
    
    
    
    if application=='plot':
        
        plt.plot(x,y)
        plt.title('----')
        plt.grid()
        plt.show()
        
    elif application=='max_attraction' :
        maxatract=y.max()
        return maxatract


#------
#telegram 
#-------
def Stress_Strain(data,application):
    stress=np.array(data['stress'])
    strain=np.array(data['strain'])
    
    if application=='plot':
        plt.plot(strain,stress)
        plt.title(-----)
        ....
        plt.show()
        
    elif application=='maxstress':
        maxstress=stress.max()
        return maxstress
    
    elif application=='maxstrain':
        maxstrain=strain.max()
        return maxstrain
    
    
    elif application=='alipilehvar':
        apm=maxstreeess+maxstrain















