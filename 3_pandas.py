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














