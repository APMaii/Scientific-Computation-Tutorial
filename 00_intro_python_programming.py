#---------NOTHIGN YET

#ipython console and Shel commands

'''
------------------ OVER VIEW --------------------------
1- Collections
2-built in functions
3-read file , csv, pdf , ..
4-time
5- functions
6- advanced class
-------------------------------------------------------
'''








'''
echo 'hello world''       #echo is lik epython's print function
pwd                     #pwd= print working directory
ls                     #list working diectory contents

cd project                 #cd -->change directory 

mkdir myproject ----->   #mkdir= make new directory
cd myproject 

mv ../myproject.text ./  #move from one directory up(../) to the current directory (./)


cd ~
cd ..
cd -




Ipython commands
you can use ls, mkdir, cd ,....
you can run your code

%xmode verbose ---> The exception reporing mdoe : Verbose --> adds some extra information

%debug --> it can get us the debugging ability 


Parital list of debugging commands

list     show the current location in the file
h(elp)  show list of command
q(uit)  quit the debugger and the program
c(ontinue)   quit the debugger, contineu in program
n(ext)    go to the next setop of the program
<enter>   repoear the previous command

p(rint)  print vairbales
s(tep)   step into subroutine
r(eturn)   retuen out of subroutine



%time    time th eexcdetution of single statement

%timeit    TIME REPEATED EXCETUTION SINGL STATEMENT FOR MROE ACCURACY
%prun     Run code with profiler
%lprun       line-by-line profiler

%memit     measure the memory use of single statement

%mprun    run code with line by ine memory profiler



'''






#-----------------------------------
#-----------ADVANCED----------------
#-----------------------------------

/n
/'
/''
//

/ #float 
// #Int

'''
Taghadom
()
**
/ // %
+ -
comparison
== !=
not
and
or 
=
'''



#for list
.sorted()
.sorted(reversed=True)
.reversed()



#for more exact instead of float we use this--->
from decimal import Decimal

principal = Decimal('1000.000')


for key,value in mydict.item():
  pass




#==================================
#-------COLLECTIONS------------------
#==================================


'''
These are the most commonly used built-in collections in Python

 A list is a mutable, ordered collection of elements, which can be of any data type.
A tuple is an immutable, ordered collection of elements.
A set is an unordered collection of unique elements. It automatically removes duplicates.
A dictionary is a collection of key-value pairs where each key must be unique.
'''


#------------------
'''
For multiple variables we can use iterations
but eahc one has ddifferent thing

str --> ucnhanged --> you can access by index buy you can not chnage that 
list --> changable, ordered , allow duplicated --> so you can access by index and change and also ordereed
tuple--> unchangable , ordered , allow duplicated --> you can acces by index but not changed --. for fdata base and .. ()
set -> unchangable, unoirdered, not duplicated--> this is {} and unchangabel but no idnex and order and it can use when you want to remoev duplciated
dictionary --> it is liek list but instead of index youc an use keys --> so in indexing youc an access wiuth tehem

'''
#----LIST--------
#emali
#.insert(index,elemnt)
#.append()
#.remove()
#.pop()
#.clear()
#.view() 
#.copy()


#---Tuple---
#unchangable 
#a[0]=.. not
#print(my_tuple.count(2))  # 1

# Find index of element
#print(my_tuple.index(3))  # 2

#--Dictionary---
#instead of index --> key and value
#a.items()
#a.values()
#a.keys()
#a['new']='new_value'

my_dict.pop('name')

# Keys, values, items
print(my_dict.keys())   # dict_keys(['age'])
print(my_dict.values()) # dict_values([30])



#-----sets---------

#---ADD-----
A = {1, 2, 3}
A.add(4)
print(A)  # Output: {1, 2, 3, 4}
#----remove----
A = {1, 2, 3}
A.remove(2)
print(A)  # Output: {1, 3}
A.discard(5)  # No error even if 5 is missing
#clear--
A = {1, 2, 3}
A.clear()
print(A)  # Output: set()




#---Union (ejtema)------

A = {1, 2, 3}
B = {3, 4, 5}
# Using `|` operator
print(A | B)  # Output: {1, 2, 3, 4, 5}
# Using `union()`
print(A.union(B))  # Output: {1, 2, 3, 4, 5}



#Intersection (eshterak)-----
A = {1, 2, 3}
B = {3, 4, 5}
# Using `&` operator
print(A & B)  # Output: {3}
# Using `intersection()`
print(A.intersection(B))  # Output: {3}




#Difference 
A = {1, 2, 3}
B = {3, 4, 5}
# Using `-` operator
print(A - B)  # Output: {1, 2}  (Elements in A but not in B)
# Using `difference()`
print(A.difference(B))  # Output: {1, 2}


#symmetric difference---
A = {1, 2, 3}
B = {3, 4, 5}
# Using `^` operator
print(A ^ B)  # Output: {1, 2, 4, 5}
# Using `symmetric_difference()`
print(A.symmetric_difference(B))  # Output: {1, 2, 4, 5}


#checking disjoint--
A = {1, 2, 3}
B = {4, 5, 6}
C = {3, 4, 5}
print(A.isdisjoint(B))  # Output: True (No common elements)
print(A.isdisjoint(C))  # Output: False (3 is common)


#checking upset
#Subset (<= or .issubset()) → Checks if all elements of a set exist in another set.
#Superset (>= or .issuperset()) → Checks if a set contains all elements of another set.
A = {1, 2}
B = {1, 2, 3, 4}
print(A <= B)  # Output: True (A is a subset of B)
print(B >= A)  # Output: True (B is a superset of A)
print(A.issubset(B))  # Output: True
print(B.issuperset(A))  # Output: True






#-----Advanced Python Collections (from collections module)-----------

#---------
# defaultdict
#----------
from collections import defaultdict
# Create a defaultdict with list as the default factory
d = defaultdict(list)
# Append values to a key
d['fruit'].append('apple')
d['fruit'].append('banana')
# No key error, it will create an empty list by default
print(d['fruit'])  # ['apple', 'banana']
print(d['vegetable'])  # []





#--------
#OrderedDict: Dictionary with Order Guarantee
#---------
#An OrderedDict maintains the order of key insertion.
from collections import OrderedDict
od = OrderedDict()
od["a"] = 1
od["b"] = 2
od["c"] = 3
print(od)  # OrderedDict([('a', 1), ('b', 2), ('c', 3)])


#reordering
od.move_to_end("a")  # Moves 'a' to the end
print(od)  # OrderedDict([('b', 2), ('c', 3), ('a', 1)])

od.move_to_end("c", last=False)  # Moves 'c' to the beginning
print(od)  # OrderedDict([('c', 3), ('b', 2), ('a', 1)])






#-----------
#Counter: Counting Hashable Objects
#------------
from collections import Counter
items = ["apple", "banana", "apple", "orange", "banana", "apple"]
counter = Counter(items)
print(counter)  
# Output: Counter({'apple': 3, 'banana': 2, 'orange': 1})



print(counter.most_common(1))  # [('apple', 3)]
print(counter["banana"])  # 2 (returns 0 if key is missing)
counter.update(["apple", "apple"])
print(counter)  
# Output: Counter({'apple': 5, 'banana': 2, 'orange': 1})


counter1 = Counter("aabbcc")
counter2 = Counter("abc")
print(counter1 + counter2)  # Counter({'a': 3, 'b': 3, 'c': 3})
print(counter1 - counter2)  # Counter({'a': 1, 'b': 1, 'c': 1})


#or it is a type
from collections import Counter
# Create a Counter from a list
count = Counter(['apple', 'orange', 'apple', 'apple', 'orange', 'banana'])
# Access counts of elements
print(count['apple'])  # 3
print(count['orange'])  # 2
# Most common elements
print(count.most_common(1))  # [('apple', 3)]



#-----------
#A deque (double-ended queue) is a list-like container optimized for fast appends and pops from both ends.
#-------
from collections import deque

# Create a deque
d = deque([1, 2, 3])
# Append to the right
d.append(4)
# Append to the left
d.appendleft(0)
# Pop from the right
d.pop()
# Pop from the left
d.popleft()
print(d)  # deque([1, 2, 3])


#--rotate and extent
dq.rotate(1)  # Shift elements right
print(dq)  # deque(['c', 'a', 'b'])

dq.rotate(-1)  # Shift elements left
print(dq)  # deque(['a', 'b', 'c'])



#-------
#A namedtuple is a subclass of tuple with named fields for better readability and access.
#--------
from collections import namedtuple

# Define a namedtuple for a point in 2D space
Point = namedtuple('Point', ['x', 'y'])

# Create a Point object
p = Point(10, 20)

# Access by field name
print(p.x, p.y)  # 10 20

# Access by index
print(p[0], p[1])  # 10 20




#------
#specialized : ChainMap
#-------
#A ChainMap groups multiple dictionaries together to create a single view.

from collections import ChainMap

# Create multiple dictionaries
dict1 = {'x': 1, 'y': 2}
dict2 = {'y': 3, 'z': 4}

# Create a ChainMap
chain = ChainMap(dict1, dict2)

# Access values (looks in dict1 first, then dict2)
print(chain['y'])  # 2 (from dict1)
print(chain['z'])  # 4 (from dict2)









#=========================================================
#========ADVANCE PYTHON BUILT IN FUCNTION================

all([True,True,True,True]) #True
all([4>2,4>1,5>1,5!=5]) #False
any(([True,True,True,True])) # True
#-------


name = "Alice"
age = 25
formatted_string = "My name is {} and I am {} years old.".format(name, age)
print(formatted_string)


formatted_string = "My name is {0} and I am {1} years old.".format("Alice", 25)
print(formatted_string)


pi = 3.14159
print("Pi is {:.2f}".format(pi))  # Output: Pi is 3.14


print("{:<10}".format("left"))   # Left-align within 10 spaces
print("{:>10}".format("right"))  # Right-align within 10 spaces
print("{:^10}".format("center")) # Center within 10 spaces



f'{year:>2}{amount:>10.2f}'
f'{number:.2f}'
f'{number:d}'


num = 1000000
print("{:,}".format(num))  # Output: 1,000,000


name = "Alice"
age = 25
print(f"My name is {name} and I am {age} years old.")


#-------------------
#.join() reverse of .split()
#.partition(:)


id(x) #---> address


a=['ali','vahid','reza']
for i in enumerate(a):
    print(i)
'''
(0, 'ali')
(1, 'vahid')
(2, 'reza')


'''




for index,value in enumerate(a):
    #hbarkari delet khas mitoni koni
    #skahtane data
    #BA TARTIB , SHOAMRE ROW
    print('shoamereye',index,' : ',value)




#----ZIP , COMBINE BOTH
#-----zip() ciombgine
a=['ali','hamid','reza']
b=[20,30,40]

#beham bechasbonm 
#numpy --> az tabe hash edstefade
#pandas --.az tabesha

#khdoe python 
combined=zip(a,b)

for aval,dovom in combined:
    print(aval,dovom)





my_professional=[i for i in myusers if i.count('a')!=0 ]
my_final={ i:j   for i,j in zip(myusers,current_balance)   }



[item for item in numbers if is_odd(item)]










lambda arguments: expression

add = lambda x, y: x + y
print(add(2, 3))  # Output: 5

x= lambda a: a*10
x(20) # 200
zarb_dar_dah=lambda a: a*10
zarb_dar_dah(20)

def myfunc():
    #bejaye return
    yield 10
    yield 20
    yield 30
    
a=myfunc()
print(a)

for i in a:
    print(i)


#-----
filter(function, iterable)

nums = [1, 2, 3, 4, 5, 6]
even = filter(lambda x: x % 2 == 0, nums)
print(list(even))  # Output: [2, 4, 6]


#----
map(function, iterable)
nums = [1, 2, 3]
squared = map(lambda x: x**2, nums)
print(list(squared))  # Output: [1, 4, 9]




list(filter(lambda x :x%2 !=0,numbers))
list(map(lambda x:x**2,numbers))

#map and filter
list(map(lambda x:x**2, filter(lambda x:x%2!=0,numbers)))









#-----TRY / EXCEPT
try:
    print('salam')
    a=b+1
except:
    print('nothing happend')


try:
    x = 5 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")  



#Occurs when a function receives an argument of the right type but an inappropriate value.
try:
    num = int("hello")  # Cannot convert "hello" to an integer
except ValueError:
    print("Invalid input! Expected a number.")


#Occurs when an operation is applied to an object of an inappropriate type.
try:
    result = "5" + 2  # Cannot add string and integer
except TypeError:
    print("Type mismatch error!")


try:
    print(xyz)  # 'xyz' is not defined
except NameError:
    print("Variable not defined!")



try:
    lst = [1, 2, 3]
    print(lst[5])  # Index 5 does not exist
except IndexError:
    print("Index out of range!")


try:
    d = {"name": "Alice"}
    print(d["age"])  # 'age' key does not exist
except KeyError:
    print("Key not found in dictionary!")


try:
    f = open("nonexistent.txt", "r")
except FileNotFoundError:
    print("File not found!")


try:
    num = 10
    num.append(5)  # Integers do not have an `append` method
except AttributeError:
    print("Invalid attribute usage!")


try:
    import non_existing_module
except ModuleNotFoundError:
    print("Module not found!")


try:
    from math import unknown_function
except ImportError:
    print("Function does not exist in module!")


#---or multiple error
#--one option----
try:
    from math import unknown_function
except ImportError:
    print("Function does not exist in module!")
except ModuleNotFoundError:
    print("Module not found!")
else:
  print('other things')
  


#----or--------
try:
    x = int("hello")  # This raises a ValueError
except (ValueError, TypeError, ZeroDivisionError) as e:
    print(f"Error occurred: {e}")




#-----or unknown error-------
try:
    new=[]
    for i in a :
        new.append(i+b)
        #c=a[i]+b
        
    print(new[1])
except Exception as e:
  print(f'man barename ro nakahbodam ama erroret hast : {e}')


#also else and finally we have







#-----or custome raise------
if float(a)<100:
    raise ValueError('Lotfan adade bozorg tar az 100 ro vared konid')



#--------------------------------------------
#---------------FILE HANDLER-----------------
'''
when python go it has 3 file
3 standard file object and it has
sys.stding --> somehting liek inptu allow python get keyboard
sys.stout --> fpr print or anything
sys.stderr--> to show the error to users



'''

#becuase we dont want to .close() or anythign we juyst get and then in variable and processing

with open("example.txt", "r") as file:
    content = file.read()
    print(content)  # Reads the file content


#in the open() you have mode
'''
"r"	Read mode (default) - File must exist
"w"	Write mode - Overwrites existing file or creates new one
"a"	Append mode - Adds data to the file if it exists
"x"	Exclusive creation - Fails if the file already exists
"b"	Binary mode - Used for non-text files like images/PDFs
"t"	Text mode (default) - Used for text files
"r+"	Read and write mode (file must exist)
"w+"	Write and read (overwrites file)
"a+"	Append and read (adds new content without removing existing data)

'''



#after opening as object we have the read() or write()
# Write Mode ("w") - Overwrites the file
with open("example.txt", "w") as file:
    file.write("Hello, world!")

# Append Mode ("a") - Adds new content
with open("example.txt", "a") as file:
    file.write("\nThis is a new line.")

# Read and Write Mode ("r+")
with open("example.txt", "r+") as file:
    print(file.read())  # Read existing content
    file.write("\nNew content added.")  # Adds new content




#-----------------
#-----CSV----------
#-----------------
import csv
#reading------
with open("data.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)  # Prints each row as a list


#writing-----
with open("data.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Name", "Age", "City"])
    writer.writerow(["Alice", 25, "New York"])
    writer.writerow(["Bob", 30, "London"])


#reasding csv as dictionary ---------
with open("data.csv", "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        print(row["Name"], row["Age"])  # Accessing columns by header name


#writing csv as dictionary ---------
with open("data.csv", "w", newline="") as file:
    fieldnames = ["Name", "Age", "City"]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({"Name": "Charlie", "Age": 22, "City": "Paris"})





#-----------------
#-----JASON----------
#-----------------
#Writing to a JSON File
import json

data = {"name": "Alice", "age": 25, "city": "New York"}

with open("data.json", "w") as file:
    json.dump(data, file, indent=4)  # Writes JSON data with indentation




#reading
with open("data.json", "r") as file:
    data = json.load(file)
    print(data)  # Outputs a dictionary



#--------
#---Working with Image Files (PIL Module)
#---------

from PIL import Image
with Image.open("image.jpg") as img:
    img.show()  # Opens the image using the default viewer



with Image.open("image.jpg") as img:
    img_resized = img.resize((200, 200))
    img_resized.save("resized_image.jpg")




#-------
#Working with PDF Files (PyPDF2 Module)
#--------
#reading PDF FILE
import PyPDF2

with open("document.pdf", "rb") as file:
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        print(page.extract_text())  # Extracts text from each page




#writing PDF---
from PyPDF2 import PdfWriter

writer = PdfWriter()
with open("document.pdf", "rb") as file:
    reader = PyPDF2.PdfReader(file)
    writer.add_page(reader.pages[0])  # Add first page to a new PDF

with open("new_document.pdf", "wb") as new_file:
    writer.write(new_file)







#----------------------
#-------TIME----------
#----------------------
'''
time – Deals with timestamps and time-related operations.
datetime – Provides more sophisticated date and time handling, including formatting, arithmetic, and time zones.
'''

# Getting the Current Time
import time
timestamp = time.time()
print(timestamp)  # 1711034354.502318
#The time.time() function returns the current time as a floating-point number representing seconds since the epoch (January 1, 1970).

#Converting Timestamp to Readable Time
print(time.ctime(time.time()))  # "Wed Mar 20 15:39:14 2025"


#Using time.sleep() for Delays
print("Start")
time.sleep(2)  # Waits for 2 seconds
print("End")


#Getting the Current Local Time (time.localtime())
local_time = time.localtime()
print(local_time)  # time.struct_time(tm_year=2025, tm_mon=3, tm_mday=20, ...)

#Formatting Time (time.strftime())
formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print(formatted_time)  # "2025-03-20 15:45:10"

#Parsing a Time String (time.strptime())
parsed_time = time.strptime("2025-03-20 15:45:10", "%Y-%m-%d %H:%M:%S")
print(parsed_time.tm_year)  # 2025


# Measuring Execution Time (time.perf_counter())
start = time.perf_counter()
time.sleep(1.5)
end = time.perf_counter()
print(f"Execution Time: {end - start:.2f} seconds")  # "Execution Time: 1.50 seconds"








#=====
#datetime Module
#======
from datetime import datetime

now = datetime.now()
print(now)  # "2025-03-20 15:50:30.123456"



print(now.year)   # 2025
print(now.month)  # 3
print(now.day)    # 20
print(now.hour)   # 15
print(now.minute) # 50
print(now.second) # 30



#--creatng specific time
custom_date = datetime(2025, 3, 20, 14, 30, 0)
print(custom_date)  # "2025-03-20 14:30:00"



#----formatting
formatted_date = now.strftime("%Y-%m-%d %H:%M:%S")
print(formatted_date)  # "2025-03-20 15:50:30"


#parsing 
parsed_date = datetime.strptime("2025-03-20 15:50:30", "%Y-%m-%d %H:%M:%S")
print(parsed_date)  # "2025-03-20 15:50:30"



#--difference---
from datetime import timedelta

# Add 7 days
future_date = now + timedelta(days=7)
print(future_date)  # "2025-03-27 15:50:30"

# Subtract 1 hour
past_date = now - timedelta(hours=1)
print(past_date)  # "2025-03-20 14:50:30"



#----------getting difference
date1 = datetime(2025, 3, 20)
date2 = datetime(2025, 4, 5)
difference = date2 - date1
print(difference.days)  # 16




#----workign with tiemzone
from datetime import datetime
import pytz

# Get UTC time
utc_now = datetime.now(pytz.utc)
print(utc_now)  # "2025-03-20 15:50:30+00:00"

# Convert UTC to another timezone
tehran_tz = pytz.timezone("Asia/Tehran")
tehran_time = utc_now.astimezone(tehran_tz)
print(tehran_time)  # "2025-03-20 19:20:30+03:30"



timestamp = 1711034354  # Example timestamp
readable_time = datetime.fromtimestamp(timestamp)
print(readable_time)  # "2025-03-20 15:50:30"




#excutation----
import timeit
execution_time = timeit.timeit("sum(range(1000))", number=10000)
print(f"Execution Time: {execution_time:.5f} seconds")







#======================================
#-----------Notes On functions---------------
#======================================

#---type hints----
def add(a: int, b: int) -> int:
    return a + b

result = add(5, 3)  # Output: 8


def greet(name: str) -> str:
    return f"Hello, {name}!"

def divide(a: float, b: float) -> float:
    return a / b

print(greet("Ali"))  # Output: Hello, Ali!
print(divide(5.0, 2.0))  # Output: 2.5

def process_numbers(numbers: list[int]) -> list[int]:
    return [num * 2 for num in numbers]

print(process_numbers([1, 2, 3]))  # Output: [2, 4, 6]



def get_user_info() -> tuple[str, int]:
    return ("Ali", 25)

def user_data(data: dict[str, int]) -> None:
    print(data)

print(get_user_info())  # Output: ('Ali', 25)
user_data({"age": 30, "score": 100})  # Output: {'age': 30, 'score': 100}







#-----ptional and Union Types------
from typing import Optional, Union

def get_age(age: Optional[int] = None) -> Optional[int]:
    return age

print(get_age())  # Output: None
print(get_age(25))  # Output: 25




def square(value: Union[int, float]) -> Union[int, float]:
    return value * value
print(square(5))  # Output: 25
print(square(2.5))  # Output: 6.25





#Closures (Functions Inside Functions)----------
def outer_function(text: str):
    def inner_function():
        print(text)  # The inner function remembers 'text'
    return inner_function

closure = outer_function("Hello, Closure!")
closure()  # Output: Hello, Closure!





#-----args kwargs------
def sum_all(*args: int) -> int:
    return sum(args)

print(sum_all(1, 2, 3, 4))  # Output: 10




#as dictionary
def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Ali", age=25, city="Tehran")
# Output:
# name: Ali
# age: 25
# city: Tehran





#-----decorators----
def decorator(func):
    def wrapper():
        print("Before function")
        func()
        print("After function")
    return wrapper

@decorator
def say_hello():
    print("Hello!")

say_hello()
# Output:
# Before function
# Hello!
# After function




#----decorators with args----
def repeat(n: int):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(n):
                func(*args, **kwargs)
        return wrapper
    return decorator

@repeat(3)
def greet(name: str):
    print(f"Hello, {name}!")

greet("Ali")
# Output:
# Hello, Ali!
# Hello, Ali!
# Hello, Ali!





#_---generators-----
def counter():
    n = 0
    while n < 5:
        yield n
        n += 1

gen = counter()
print(next(gen))  # Output: 0
print(next(gen))  # Output: 1





#======================================
#-----------------------------------
#-----------CLASS---------------
#-----------------------------------
#======================================

'''

Object Oriented Programming (OOP_

You can download the [ackages from github, bitbucket, sourceforge.net
you can downlaod and isntall from pip and conda 
we have base class or siperclass
and also we have drived class or subclass

also we have polymorphism

for creation of object from oen class we need
constructure expression

we have class header --> class name:
also with semi columns we can have the description of the class (documentation)



'''




#--------------------------------------------------------
#-----------------    __INIT__  -------------------------
#--------------------------------------------------------



'''
initial assinment with __init__ methods
this method must return None (liek all function that ahs no none return) if not it get us Typeerror
it must has self --> to have connections  betweeen functions
any functions that has __ and __ --> we said special method



This is initiator --> it just get the values you can save them in self.
also you can save another thigns in another things (or anythign that doesn get from the user)

you can also have default --> . Default Values in __init__
def __init__(self, name="Unknown", age=18):


You can define private attributes inside __init__ using a double underscore (__).
def __init__(self, owner, balance):
        self.owner = owner
        self.__balance = balance  # Private attribute

# print(account.__balance)  # ❌ AttributeError: 'BankAccount' object has no attribute '__balance'

but you can get the access

def get_balance(self):
        return self.__balance



or alos you can have
class Dog:
    species = "Canine"  # Class attribute

    def __init__(self, name):
        self.name = name  # Instance attribute

**Class attributes belong to the class itself and are shared across instances.
Instance attributes are unique to each object.




__init__ with Type Hints
class Employee:
    def __init__(self, name: str, age: int, salary: float):
        self.name = name
        self.age = age
        self.salary = salary


#or using auomticlaly
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int

p = Person("Alice", 25)




# __init__ with Variable Arguments (*args, **kwargs)
class Car:
    def __init__(self, *args, **kwargs):
        self.brand = kwargs.get("brand", "Unknown")
        self.year = kwargs.get("year", 2020)

car1 = Car(brand="Toyota", year=2023)
print(car1.brand, car1.year)  # Output: Toyota 2023




'''




#--------------------------------------------------------
#-----------------    Encapsulation  -------------------------
#--------------------------------------------------------
'''
Acdtually we have public data and private data
'''

#Public data
#Can be accessed and modified freely from anywhere.
#No leading underscore in variable names.

class Person:
    def __init__(self, name, age):
        self.name = name  # Public attribute
        self.age = age  # Public attribute

p = Person("Alice", 30)

print(p.name)  # ✅ Accessible: Alice
p.name = "Bob"  # ✅ Modifiable
print(p.name)  # ✅ Updated: Bob



#private data
#Cannot be accessed directly from outside the class.
#Prefixed with double underscores (__).
#Python uses name mangling (_ClassName__attribute) to prevent direct access.
class BankAccount:
    def __init__(self, owner, balance):
        self.owner = owner  # Public
        self.__balance = balance  # Private

account = BankAccount("John", 1000)

print(account.owner)  # ✅ Accessible: John
# print(account.__balance)  # ❌ AttributeError: 'BankAccount' object has no attribute '__balance'




#Accessing Private Data (Workarounds)
class BankAccount:
    def __init__(self, owner, balance):
        self.__balance = balance  # Private

    def get_balance(self):  # Getter method
        return self.__balance

account = BankAccount("John", 1000)
print(account.get_balance())  # ✅ Output: 1000





# @property and @setter (Encapsulation in Python)--------------
#Python provides @property and @setter decorators for controlled attribute access.

'''
Why Use @property?
Allows getter methods to be called like attributes.
Provides read-only access without exposing private attributes.
Helps enforce data validation.
'''

class Employee:
    def __init__(self, name, salary):
        self.name = name
        self.__salary = salary  # Private

    @property
    def salary(self):
        return self.__salary  # Getter method

emp = Employee("Alice", 5000)

print(emp.salary)  # ✅ Accesses private variable as an attribute
# emp.salary = 6000  # ❌ AttributeError: Can't set attribute


#Salary is read-only because no @salary.setter is defined.

#----------------------------------
#----------------------------------
#----------------------------------

#Using @setter (Allowing Modification)
class Employee:
    def __init__(self, name, salary):
        self.name = name
        self.__salary = salary  # Private

    @property
    def salary(self):
        return self.__salary  # Getter

    @salary.setter
    def salary(self, new_salary):
        if new_salary < 0:
            raise ValueError("Salary cannot be negative!")
        self.__salary = new_salary  # Setter

emp = Employee("Bob", 5000)
print(emp.salary)  # ✅ 5000

emp.salary = 6000  # ✅ Updates salary
print(emp.salary)  # ✅ 6000

# emp.salary = -500  # ❌ ValueError: Salary cannot be negative!






#------------------------
#__repr__ vs. __str__ (Object Representation)
#--------------------------
#Python provides __repr__ and __str__ methods for defining string representations of objects.

'''
__repr__ (Official String Representation)

Goal: Return an unambiguous string representation.
Used for debugging and development.
Should return a string that could be used to recreate the object.
'''

class Car:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model

    def __repr__(self):
        return f"Car('{self.brand}', '{self.model}')"

car = Car("Toyota", "Corolla")
print(repr(car))  # ✅ Output: Car('Toyota', 'Corolla')


'''
__str__ (User-Friendly String Representation)
Goal: Return a human-readable string.
Used for display (print()).
'''
class Car:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model

    def __str__(self):
        return f"{self.brand} {self.model}"

car = Car("Toyota", "Corolla")

print(str(car))  # ✅ Output: Toyota Corolla
print(car)  # ✅ Output: Toyota Corolla (calls __str__ automatically)

#If __str__ is not defined, Python falls back to __repr__.






#@staticmethod – Independent Utility Methods---
#Does not require self (instance).
#Used when a function doesn’t depend on the instance or class.

class MathOperations:
    @staticmethod
    def add(x, y):
        return x + y

print(MathOperations.add(5, 3))  # ✅ Output: 8




#@classmethod – Operating on the Class Itself
#akes cls as the first parameter instead of self.

class Counter:
    count = 0  # Class variable

    @classmethod
    def increment(cls):
        cls.count += 1

Counter.increment()  # ✅ Works without an instance
print(Counter.count)  # ✅ Output: 1


class Counter:
    count = 0  # Class variable (shared among all instances)

    @classmethod
    def increment(cls):
        """Modifies the class-level variable."""
        cls.count += 1

# Calling the method without an instance
Counter.increment()
print(Counter.count)  # ✅ Output: 1

# Creating instances and calling the method
c1 = Counter()
c2 = Counter()
c1.increment()
print(Counter.count)  # ✅ Output: 2



#-----------------
#----Inherents----
#-----------------
class Animal:
    def speak(self):
        return "I make a sound"

class Dog(Animal):
    
    def new(self):
        return 'hi inherent'


a1=Animal()
a1.speak() #'I make a sound'

a2=Dog()
a2.speak() # 'I make a sound'
a2.new() # 'hi inherent'






#---MULTI LEVEL INHERENT----
class Grandparent:
    def greet(self):
        return "Hello from Grandparent"

class Parent(Grandparent):
    pass

class Child(Parent):
    pass

c = Child()
print(c.greet())  # ✅ Output: Hello from Grandparent







#-------------
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return "Some sound"

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)  # Calls parent constructor
        self.breed = breed

    def speak(self):
        return super().speak() + " but Woof!"  # Calls parent method

dog = Dog("Buddy", "Golden Retriever")
print(dog.name)   # ✅ Output: Buddy
print(dog.speak())  # ✅ Output: Some sound but Woof!





#--------
#override-----
class Animal:
    def speak(self):
        return "I make a sound"

class Dog(Animal):
    def speak(self):
        return "Woof!"

d = Dog()
print(d.speak())  # ✅ Output: Woof!



#--------------
#all classes inherent must have this abstract method
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def speak(self):
        pass  # Must be implemented in child class

class Dog(Animal):
    def speak(self):
        return "Woof!"

d = Dog()
print(d.speak())  # ✅ Output: Woof!




#---------
#---------
#---------
#-----------------------------
#-----------------------------
'''   DATACLASS      '''
#-----------------------------
#-----------------------------
#---------
#---------
#---------

'''
The @dataclass decorator (introduced in Python 3.7) is used to create immutable or 
mutable objects with less boilerplate code. It automatically generates methods like:

__init__
__repr__
__eq__
__hash__

'''
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __repr__(self):
        return f"Person(name={self.name}, age={self.age})"

p = Person("Alice", 30)
print(p)  # ✅ Output: Person(name=Alice, age=30)

#withd dataclass you can create automatically
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int

p = Person("Alice", 30)
print(p)  # ✅ Output: Person(name='Alice', age=30)




#---------------------------------
#---------------------------------
#with default values
@dataclass
class Employee:
    name: str
    salary: float = 50000  # Default salary

e1 = Employee("Bob")
e2 = Employee("John", 70000)

print(e1)  # ✅ Output: Employee(name='Bob', salary=50000)
print(e2)  # ✅ Output: Employee(name='John', salary=70000)









#---------------------------------
#---------------------------------
#frozne --> immutable
@dataclass(frozen=True)
class Book:
    title: str
    price: float

b = Book("Python 101", 29.99)
print(b.title)  # ✅ Output: Python 101

b.price = 39.99  # ❌ ERROR: FrozenInstanceError: cannot assign to field 'price'
#Use case: Prevent accidental modification of objects.



#---------------------------------
#---------------------------------
#filed() with advanced control
#metadata is useful for additional info like currency, measurement units, etc.

from dataclasses import field

@dataclass
class Product:
    name: str
    price: float = field(default=0.0, metadata={"unit": "USD"})  # Metadata can store extra info

p = Product("Laptop", 999.99)
print(p)  # ✅ Output: Product(name='Laptop', price=999.99)




#or also for default 
@dataclass
class Student:
    name: str
    subjects: list = []  # ❌ Bad practice! This list is shared across instances.

s1 = Student("Alice")
s1.subjects.append("Math")

s2 = Student("Bob")
print(s2.subjects)  # ❌ Unexpected output: ['Math']


@dataclass
class Student:
    name: str
    subjects: list = field(default_factory=list)  # ✅ Creates a new list per instance

s1 = Student("Alice")
s1.subjects.append("Math")

s2 = Student("Bob")
print(s2.subjects)  # ✅ Output: []





#---------------------------------
#---------------------------------
#Dataclasses don’t support ordering (<, >, <=, >=) by default. To enable it:
@dataclass(order=True)
class Player:
    score: int

p1 = Player(10)
p2 = Player(20)

print(p1 < p2)  # ✅ Output: True





#---------------------------------
#---------------------------------
# Useful for serialization (e.g., saving as JSON).
#convert to dict
from dataclasses import asdict, astuple

@dataclass
class User:
    username: str
    email: str

u = User("john_doe", "john@example.com")

print(asdict(u))  # ✅ Output: {'username': 'john_doe', 'email': 'john@example.com'}
print(astuple(u))  # ✅ Output: ('john_doe', 'john@example.com')


import json

json_data = json.dumps(asdict(p))  # ✅ {'name': 'Laptop', 'price': 1200.99}
print(json_data)

# Convert back to object
p_dict = json.loads(json_data)
p_obj = Product(**p_dict)
print(p_obj)  # ✅ Output: Product(name='Laptop', price=1200.99)

