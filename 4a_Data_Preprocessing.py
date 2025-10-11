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

















