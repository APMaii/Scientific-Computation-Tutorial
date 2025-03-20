#---------NOTHIGN YET



#ipython console and Shel commands

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







#------------------
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


#--Dictionary---
#instead of index --> key and value
#a.items()
#a.values()
#a.keys()
#a['new']='new_value'



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







#==================================
#-------COLLECTIONS------------------
#==================================








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
list(filter(lambda x :x%2 !=0,numbers))
list(map(lambda x:x**2,numbers))

#map and filter
list(map(lambda x:x**2, filter(lambda x:x%2!=0,numbers)))




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





#-----TRY / EXCEPT
try:
    print('salam')
    a=b+1
except:
    print('nothing happend')



try:
    new=[]
    for i in a :
        new.append(i+b)
        #c=a[i]+b
        
    print(new[1])
except Exception as e:
  print(f'man barename ro nakahbodam ama erroret hast : {e}')






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










#======================================
#-----------------------------------
#-----------CLASS---------------
#-----------------------------------
#======================================

