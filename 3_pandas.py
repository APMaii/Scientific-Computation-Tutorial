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

#remove----------
birthyears = people.pop("birthyear")
del people["children"]


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



















