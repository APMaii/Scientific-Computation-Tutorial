'''

Author : Ali Pilehvar Meibody



Actualy in python , when you assign variables like int or float or ..
it just write the name of variable and the value
but in other low level programming languages liek C or Cpp you must write 
type name = value
and it store only in byte or bytes 

so but in python this is C object that store, for instance it must has the name , size, type and the value
so it is really memory consuming and also object in the meemory.

so for the list we have also more memory consumers liek a lot of eleemnts and its type and..
so we need more efficient way


'''


#-------------------------------------------------------
#-------------------INTRODUCTION------------------------
#-------------------------------------------------------

import numpy as np 

a1=list((1,2,3,4))
a2=numpy.array([1,2,3,4])
a2=numpy.array((1,2,3,4))

a2=numpy.array((10,'hamid',30,True))
type(a2) #Out[35]: numpy.ndarray

#----0D-----> fght ye adad noghte
a=numpy.array(12)
a.ndim #Out[49]: 0
a.size #Out[55]: 1 yani ydone toosh
a.shape #Out[61]: ()
print("itemsize:", a.itemsize, "bytes")
print("nbytes:", a.nbytes, "bytes")



#----1D----> liste ye bodi ye  khate
b=list((10,20,30))
b=[10,20,30]
a=numpy.array(b)
a=numpy.array((10,20,30,40,50,60,70))
a=numpy.array([10,20,30,40,50,60,70]) #standard
a[0]
a[1]
a[4:5:1]

#dhange?
a[0]=10



#----2D------> YEK SAFE , do ta bod
#COLUMN --> SOTOOON
#ROW ---> RADIF
a=numpy.array([ [10,20,30] ,[50,60,70] ,[87,92,100]   ])
b=numpy.array([ [10,20,30] ,[50,60,70] ])
#radif , sotoon
a[1,1]  #dar a , radife indexe 1 , sotone indexe 1 #Out[87]: 60
a[:,3]


#----3D----> a lot of 2d on each others
d=np.array([[ [10,20,30],[40,50,60]   ],[ [70,80,90],[100,110,120]   ]])
d[0,1,0] 
#aval kodom sfe, bad kodoom radif va....





#-------------------------------------------------------
#------------------Other assignments----------------
#-------------------------------------------------------

np.array([1, 2, 3, 4], dtype='float32')
#most of them need size and dtype
np.arange(0, 20, 2)
np.linspace(0, 1, 5)
np.logspace(0, 10, 10, base=e)



np.zeros(10, dtype=int)
np.empty((2,3))
np.ones((3, 5), dtype=float)
np.full((3, 5), 3.14)
np.eye(3) #identity
np.diag([1,2,3]) # a diagonal matrix
np.diag([1,2,3],k=1) # a diagonal matrix with offset

data = np.genfromtxt('stockholm_td_adj.dat') #from dat and csv


np.savetxt("random-matrix.csv", our_array)
np.load("random-matrix.npy")





#betweeen 0 and 1
np.random.random((3, 3))
#size and the things needed in randomness liek alpha and a and b
#discrete
np.random.normal(0, 1, (3, 3))
np.random.uniform(0,10,(3,3))

#integer
np.random.randint(0, 10, (3, 3))

x, y =  np.mgrid[0:5, 0:5] # similar to meshgrid in MATLAB
# x--> [000000/111111/..../555555]
#y---> [0 1 2 3 4 / 

'''
DTYPE------
bool_ Boolean astored as a byte
int_   default integer type ( same as C long: normally int64 or int32)
intc    identical to C int (normally int32 or int64)
intp    integer used for indexing ( C ssize_t , int32 or int64)
int8    byte(-128 , 127)
int16   integer ( -32768 to 32767)
int 32
int64

unit8    unsigned integer (0 to 255)
unit16 
unit32
unit64

float_   shoryhand for float64
float16
float32 
float 64

complex_ shorand for complex128
complex64
complex128
'''






#-------------------------------------------------------
#-------------------MAGIC FUNCTIONS-------------------
#-------------------------------------------------------

#-----reshaping
grid = np.arange(1, 10).reshape((3, 3))

#------type casting
M.dtype
M2 = M.astype(float)
M3 = M.astype(bool)

#---flatten
B = A.flatten()


#----copy and view
arr2=arr.view()
arr2=arr.copy()

#---fancy indexing
row_indices = [1, 2, 3]
A[row_indices]


#mask
row_mask = np.array([True, False, True, False, False])
#Or
row_mask = np.array([1,0,1,0,0], dtype=bool)

B[row_mask]


#more professional
#or
row_mask = (5 < x) * (x < 7.5)
x[row_mask]



x = np.array([1, 2, 3, 4, 5])
x < 3 # less than
#Out[5]: array([ True, True, False, False, False], dtype=bool)
x > 3 # greater than
#Out[6]: array([False, False, False, True, True], dtype=bool)
x <= 3 # less than or equal
#Out[7]: array([ True, True, True, False, False], dtype=bool)
x >= 3 # greater than or equal
#Out[8]: array([False, False, True, True, True], dtype=bool)
x != 3 # not equal
#Out[9]: array([ True, True, False, True, True], dtype=bool)
x == 3 # equal
#Out[10]: array([False, False, True, False, False], dtype=bool)



#where-->it get us the index
indices = np.where(mask)
new=np.where(a==2)
new=np.where(a==5)
new=np.where(a>30)
#wec an also get the elemnts simply
x[indices] 




#----counting--------
np.count_nonzero(x < 6)
np.sum(x < 6)
np.sum(x < 6, axis=1)

#boolean operators
np.sum((inches > 0.5) & (inches < 1))
np.sum(~( (inches <= 0.5) | (inches >= 1) ))








#----conditions
if (M > 5).any():
    print("at least one element in M is larger than 5")
else:
    print("no element in M is larger than 5")




if (M > 5).all():
    print("all elements in M are larger than 5")
else:
    print("all elements in M are not larger than 5")
  





#-------------------------------------------------------
#-------------JOINING ANS SPLITTING---------------------
#-------------------------------------------------------

#*** concatenate --> for all things the all o fthem in the rows ( axis=0 )
#axis=1 --> in the columns 


a=np.array([10,20,30,40])
b=np.array([100,200,300,400])

c=np.concatenate([a,b])
#[ 10  20  30  40 100 200 300 400]




#if we have 2D 
#we can have the all in oen rows
a=np.array([10,20,30,40]).reshape(-1,1)
b=np.array([100,200,300,400]).reshape(-1,1)
c=np.concatenate([a,b])

#[[ 10]
# [ 20]
# [ 30]
# [ 40]
# [100]
# [200]
# [300]
# [400]]

c=np.concatenate([a,b],axis=1)
#10 100
#20  200
#30  300
#40  400



#Or in one column
a=np.array([10,20,30,40]).reshape(1,-1)
b=np.array([100,200,300,400]).reshape(1,-1)
c=np.concatenate([a,b])

#[[ 10  20  30  40]
# [100 200 300 400]]

c=np.concatenate([a,b],axis=1)
#[[ 10  20  30  40 100 200 300 400]]


#----------------------------
#For working with arrays of mixed dimensions , clearer to use np.vstack (vertical )
#and np.hstack (horizontal
x = np.array([1, 2, 3])
grid = np.array([[9, 8, 7], [6, 5, 4]])
np.vstack([x, grid])


y = np.array([[99],[99]])
np.hstack([grid, y])

np.stack()
np.dstack()


#----tile and repeat
a = np.array([[1, 2], [3, 4]])
np.repeat(a, 3) #array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
np.tile(a, 3) #array([[1, 2, 1, 2, 1, 2], [3, 4, 3, 4, 3, 4]])



#---splitting arrays-------
x = [1, 2, 3, 99, 99, 3, 2, 1]
x1, x2, x3 = np.split(x, [3, 5])

np.split(x,[3,5])
np.vsplit(grid, [2])
np.hsplit(grid, [2])






#-------------------------------------------------------
#-------------Iterating over array elements------------
#-------------------------------------------------------
#first row and then column


#---for 1D----
v = array([1,2,3,4])

for element in v:
    print(element)

#---for 2D----
M = array([[1,2], [3,4]])

for row in M:
    print(("row", row))
    
    for element in row:
        print(element)


#----with indexes----
for row_idx, row in enumerate(M):
    print(("row_idx", row_idx, "row", row))
    
    for col_idx, element in enumerate(row):
        print(("col_idx", col_idx, "element", element))
       
        # update the matrix M: square each element
        M[row_idx, col_idx] = element ** 2



'''
Computation on NumPy arrays can be very fast, or it can be very slow. The key to
making it fast is to use vectorized operations, generally implemented through Num‐
Py’s universal functions (ufuncs). This section motivates the need for NumPy’s ufuncs,
which can be used to make repeated calculations on array elements much more efficient.
It then introduces many of the most common and useful arithmetic ufuncs
available in the NumPy package.



The Slowness of loops
Python’s default implementation (known as CPython) does some operations very
slowly. This is in part due to the dynamic, interpreted nature of the language: the fact
that types are flexible, so that sequences of operations cannot be compiled down to
efficient machine code as in languages like C and Fortran. Recently there have been
various attempts to address this weakness: well-known examples are the PyPy project,
a just-in-time compiled implementation of Python; the Cython project, which converts
Python code to compilable C code; and the Numba project, which converts
snippets of Python code to fast LLVM bytecode. Each of these has its strengths and
weaknesses, but it is safe to say that none of the three approaches has yet surpassed
the reach and popularity of the standard CPython engine.



Main Drawback
The relative sluggishness of Python generally manifests itself in situations where
many small operations are being repeated—for instance, looping over arrays to operate on each element. For example, imagine we have an array of values and we’d like to
compute the reciprocal of each. A straightforward approach might look like this:


1 loop, best of 3: 2.91 s per loop

It takes several seconds to compute these million operations and to store the result!

but the type-checking
and function dispatches that CPython must do at each cycle of the loop. Each time
the reciprocal is computed, Python first examines the object’s type and does a
dynamic lookup of the correct function to use for that type. If we were working in
compiled code instead, this type specification would be known before the code executes
and the result could be computed much more efficiently



For many types of operations, NumPy provides a convenient interface into just this
kind of statically typed, compiled routine. This is known as a --vectorized operation--
You can accomplish this by simply performing an operation on the array, which will
then be applied to each element. This vectorized approach is designed to push the
loop into the compiled layer that underlies NumPy, leading to much faster execution.


100 loops, best of 3: 4.6 ms per loop


so we have Numpy's UFuncs
that has two flavos
Unaryufuncs --> operates on a single input
binary Ufunc --> operate on two inputs

'''

#---Array arithemic
x = np.arange(4)
print("x =", x)
print("x + 5 =", x + 5)
print("x - 5 =", x - 5)
print("x * 2 =", x * 2)
print("x / 2 =", x / 2)
print("x // 2 =", x // 2) # floor division
print("-x = ", -x)
print("x ** 2 = ", x ** 2)
print("x % 2 = ", x % 2)

np.add() #+
np.substract() #-
np.negative() #-
np.multiply() #*
np.divide() #/
np.floor_divide() #//
np.power() #**
np.mod() #%
np.absolute()
#or
np.abs()


#---comparison --> it get us teh boolean one
np.equal() #==
np.not_equal() #!=
np.less() #<
np.less_equal() #<=
np.greater() #>
np.greater_equal() #>=



#--bolean operators
np.bitwise_and() #&
np.bitwise_or() #|
np.bitwise_xor() #^
np.bitwise_no() #~




#---Trigonometric functions
theta = np.linspace(0, np.pi, 3)

print("theta = ", theta)
print("sin(theta) = ", np.sin(theta))
print("cos(theta) = ", np.cos(theta))
print("tan(theta) = ", np.tan(theta))

x = [-1, 0, 1]
print("x = ", x)
print("arcsin(x) = ", np.arcsin(x))
print("arccos(x) = ", np.arccos(x))
print("arctan(x) = ", np.arctan(x))

#Exponents and logarithms
x = [1, 2, 3]
print("x =", x)
print("e^x =", np.exp(x))
print("2^x =", np.exp2(x))
print("3^x =", np.power(3, x))

x = [1, 2, 4, 10]
print("x =", x)
print("ln(x) =", np.log(x))
print("log2(x) =", np.log2(x))
print("log10(x) =", np.log10(x))

x = [0, 0.001, 0.01, 0.1]
print("exp(x) - 1 =", np.expm1(x))
print("log(1 + x) =", np.log1p(x))


'''
NumPy has many more ufuncs available, including hyperbolic trig functions, bitwise
arithmetic, comparison operators, conversions from radians to degrees, rounding and
remainders, and much more. A look through the NumPy documentation reveals a lot
of interesting functionality.
'''
new=np.floor(a) #b paeen gerd mikone
new=np.ceil(a) #b bala



#Specialized ufuncs
#from scipy import special
x = [1, 5, 10]
print("gamma(x) =", special.gamma(x))
print("ln|gamma(x)| =", special.gammaln(x))
print("beta(x, 2) =", special.beta(x, 2))
x = np.array([0, 0.3, 0.7, 1.0])
print("erf(x) =", special.erf(x))
print("erfc(x) =", special.erfc(x))
print("erfinv(x) =", special.erfinv(x))




#========Advanced Ufunc Features========
#Specifying output
'''
For large calculations, it is sometimes useful to be able to specify the array where the
result of the calculation will be stored. Rather than creating a temporary array, you
can use this to write computation results directly to the memory location where you’d
like them to be. For all ufuncs, you can do this using the out argument of the
function:
'''
x = np.arange(5)
y = np.empty(5)
np.multiply(x, 10, out=y)
print(y)

y = np.zeros(10)
np.power(2, x, out=y[::2])
print(y)

#Aggregates
x = np.arange(1, 6)
np.add.reduce(x)
#15

np.multiply.reduce(x)
#120

#if you want to have all intermediate results
np.add.accumulate(x)
##array([ 1, 3, 6, 10, 15])

np.multiply.accumulate(x)
#array([ 1, 2, 6, 24, 120])


#========================
#-----statistics--------
#just think that we can get for all of them
#we can also get for some specific columsn or rows with axis
L = np.random.random(100)

np.sum(L)
'''
it is better sum
10 loops, best of 3: 104 ms per loop
1000 loops, best of 3: 442 μs per loop

'''

np.min(big_array)
np.max(big_array)

#or
big_array.min(), big_array.max(), big_array.sum()

M.min(axis=0)
M.max(axis=1)

np.var() #sum of elements
np.prod() #product of elemnts
np.std()  # compute standad deviation
np.var()  #variance
np.min()
np.max()
np.argmin() #find the index of min
np.argmax() #find the index o max
np.mean()
np.median()
np.percentile()
np.any()
np.all()

#also if you want to ignoring missing values so we can handle the Nan just with adding nin
#Like
np.nansum()
np.nanprod()
np.nanstd()
#.....


#********
#so we learn how we can use universal functions can be used to vectoriez operations and therby remove
#slow python loops
#another means of vectorizing operations is use numpy' broadcasting functionally

'''
Broadcasting is simply a
set of rules for applying binary ufuncs (addition, subtraction, multiplication, etc.) on
arrays of different sizes.

'''

a = np.array([0, 1, 2])
b = np.array([5, 5, 5])
a + b

#if it hasnot it get aritifical one


'''
----Rules of Broadcasting-----
Broadcasting in NumPy follows a strict set of rules to determine the interaction
between the two arrays:

• Rule 1: If the two arrays differ in their number of dimensions, the shape of the
one with fewer dimensions is padded with ones on its leading (left) side.
• Rule 2: If the shape of the two arrays does not match in any dimension, the array
with shape equal to 1 in that dimension is stretched to match the other shape.
• Rule 3: If in any dimension the sizes disagree and neither is equal to 1, an error is
raised.







'''

np.sort() #fast sorting and get bakc that
np.argsort() #it get back the index of that

#laos row and others
np.sort(X, axis=0)

#**
#Creating Structured Arrays

np.dtype({'names':('name', 'age', 'weight'),
'formats':('U10', 'i4', 'f8')})

np.dtype({'names':('name', 'age', 'weight'),
'formats':((np.str_, 10), int, np.float32)})


#in np.dtype() here we cna write these types
'''

b Byte
i  signed integer == np.int32
u unsigned ineger == np.unit8
f floating point == np.int64
c complex floating point == np.complex128
S or a  --> string
U unicode string np.str_
V raw data (void np.void)




'''




#================================
#================================
#================================
#-----Go on Matrix()--------
#================================
#================================
#================================

#most of them is has row and columns --> ndim is 2D
#and so the all the artibary addition and all of them can be applied for all of them#
#FIRST IN ASSIGNED we can use the eye() for identity 


#other things
np.zeros((3,3))
np.ones((3,3))
np.eyes((3,3))

#extract the diagonal
np.diag([1,2,3])

#it getus
#[1 0 0]
#[0 2 0 ]
#[0 0 3]



#we can use all the + - ,... * and **
#But bec arefull

A * A # element-wise multiplication
#or
np.multiply() 


'''
What about matrix mutiplication? There are two ways. We can either use the dot function,
which applies a matrix-matrix, matrix-vector, or inner vector multiplication to its two
arguments:

'''

np.dot(A, A)

#alternatively
M = matrix(A)
v = matrix(v1).T # make it a column vector

M * v

# inner product
v.T * v



#Above we have used the .T to transpose the matrix object v.
#We could also have used the transpose function to accomplish the same thing.


C = matrix([[1j, 2j], [3j, 4j]])
#transpose the matrix
C.T

#conjugate the matrix
np.conjugate(C)


#Hermitian conjugate: transpose + conjugate
C.H

#we can also extract real and other thigns
np.real(C) # same as: C.real
np.imag(C) # same as: C.imag



#complex argument and absolute value
np.angle(C+1) #matlab users
np.abs(C)


#----Inverse----
np.linalg.inv(C) # equivalent to C.I 
C.I * C

#--Determinant---
np.linalg.det(C)

np.linalg.det(C.I)












#--------------------------------------
#----------more information------------

#http://numpy.scipy.org
#http://scipy.org/Tentative_NumPy_Tutorial
#http://scipy.org/NumPy_for_Matlab_Users













