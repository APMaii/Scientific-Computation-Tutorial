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


#============================
#Other assignments-----------
np.array([1, 2, 3, 4], dtype='float32')
#most of them need size and dtype
np.arange(0, 20, 2)
np.linspace(0, 1, 5)

np.zeros(10, dtype=int)
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



#-------------------MAGIC FUNCTIONS-------------------
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


#where-->it get us the index
indices = np.where(mask)
#wec an also get the elemnts simply
x[indices] 


#extract the diagonal
np.diag(A)


#-----reshaping
grid = np.arange(1, 10).reshape((3, 3))

#------type casting
M.dtype
M2 = M.astype(float)
M3 = M.astype(bool)




#-----Iterating over array elements-------

















