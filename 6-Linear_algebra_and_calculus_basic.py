'''

Before  going for Machine learning, Linear Algebra and calculus is important
here is only overview and very very brief summary of tehse two references
https://github.com/ageron/handson-ml2/blob/master/math_linear_algebra.ipynb
https://github.com/ageron/handson-ml2/blob/master/math_differential_calculus.ipynb

'''


#================================
#================================
#================================
'''       Linear Algebra      '''
#================================
#================================
#================================



'''
Linear Algebra is the branch of mathematics that studies vector spaces and linear transformations between vector spaces, 
such as rotating a shape, scaling it up or down, translating it (ie. moving it), etc.

Machine Learning relies heavily on Linear Algebra, so it is essential to understand what vectors and matrices are, what 
operations you can perform with them, and how they can be useful.


------VECTORS------
quantity defined by magnitude and direction. rocket veloctiy is 3 dimensional vector and
magnitude is speed of rocket and its direction. vector can be represnted by an array of numbers called scalars.
each scalar --> magnitude of vector with regards to each dimension.

for instance rocket with 5000 m/s uop , 10 m/s and..

Vector --> [10 50 50000]

so what is use of vctor in Machine learning? --> actually 
for example , we built  a machine learning system to classify video inti 3 categories (good,spam,clickbait)
based on what we know about them. for each video, we would have a vector representing what we know
about it such as 
Video = [10.5 5.2 3.25 7.0) 
for instance vdieo lasts 10.5 minutes but only 5.2% viewers what more than and such informations

so in the other hand ML may predict --> class_probabilities [0.8 0.18 0.02 ) which is 80% is spam
18% clickable , 2% godo video


In python we have vector and reperesent in many ways.
simplest is regular python list
'''
a=list([10.5, 5.2, 3.25, 7.0])
b=[10.5, 5.2, 3.25, 7.0]

#or for scientific calcuyklation much betetr to use
import numpy as np
video = np.array([10.5, 5.2, 3.25, 7.0])
#size of vector -->
video.size

#th i th element (called entry or item) of vector 'v' is noted 'vi'
#video[2]  # 3rd element
#----PLOTTING
#0D VECTOR ----> JUST NUMBER 
#1D VECTOR --> JUST ONE POINT
#2D VECTORS --------
u = np.array([2, 5])
v = np.array([3, 1])

#ALSO WE CAN CREATE one vector for u , one for v
#3D VECTOR ----
#In space
a = np.array([1, 2, 8])
b = np.array([5, 6, 3]

#---------------------
'''
Norm
the norm of vector (u) noted ||u|| is measure of length(magnitude) of u
we have a lot of norm but most common is Euclidian Norm

|| u|| = radical [  Zigma ui**2 ]
'''
import numpy.linalg as LA
LA.norm(u)

#-----Addition
'''
Vectors of same size can be added together . elementwise
 [2 5]
+ [3 1]
----------
array([5, 6])


Vector addition is commutative --> u + v = v + u
Vector addition is also associative --> u + (v+w) = (u +v) + w

also if you have shape (number of vectors) you add vector V --> whoel shape
shifted by v


Multiplication by a scalar -- > scale of figure

also we have commutative --> landa * u = u * landa
associative --> landa1 * (landa2 * u) = (landa1 * landa2) * u
also it is distributuve --> landa * (u+v) = landa * u + landa * v

'''


#-------
'''
ZARO VECTOR --> FULL OF 0 
UNIT VECTOR --> NORM EQUAL TO 1
NORMALIZED VECTOR --> U^ = u / \\u\\

'''

#-----
'''
Dot product
The dot product (also called scalar product or inner product in the context of the 
Euclidian space)

U.V = ||U|| X ||V|| X COS(TETA)

U.V = ZIGMA UIX VI


product is commutative  u.v = v.u
not associative (u.v).w /= u.(v.w)
it is distributive --> u.(v+w) = u.v + u.w
and also it is only between two vectors , not scaler and vectors

'''
np.dot(u,v)
#11

#or
u.dot(v)
#11


#--prokjecting point nto an axis
#projection of vector v onto u's axis
#proju V = ( u.v / ||u||**2 ) * u
#or
#proju V = (v.u^) * u^


#===================
'''Materices'''
#===================


'''
A matrix is a rectangular array of scalars (ie. any number: integer, real or 
complex) arranged in rows and columns, for example:
'''
a=[
    [10, 20, 30],
    [40, 50, 60]
]

A = np.array([
    [10,20,30],
    [40,50,60]
])

A.shape
#(2, 3)

A.size
#6

#--element indexing --> Xi,j --> ith row , jth column
# Mi,* --> it means all columns
A[1, :]  # 2nd row vector (as a 1D array)



#-----
'''
A square matrix is a matrix that has the same number of rows and columns


An upper triangular matrix is a special kind of square matrix where all the elements 
below the main diagonal (top-left to bottom-right) are zero


Similarly, a lower triangular matrix is a square matrix where all elements above the main
diagonal are zero

A triangular matrix is one that is either lower triangular or upper triangular.

A matrix that is both upper and lower triangular is called a diagonal matrix



'''
np.diag([4, 5, 6])

'''

array([[4, 0, 0],
       [0, 5, 0],
       [0, 0, 6]])
'''



#identity matix --> diagonam matrix of n X N 1 in main dioganl
np.eye(3)

'''

array([[ 1.,  0.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.]])
'''



#------------
#adding matrices
A = np.array([
    [10,20,30],
    [40,50,60]
])

B = np.array([[1,2,3],
              [4, 5, 6]])


'''

array([[11, 22, 33],
[44, 55, 66]])
'''

'''
adding is commutative --> A + B = B+A
is associatiev --> A+ (B+C) = (A+B) + C

Scalar multipilication --> landa elemntwise to all of the elemnts
M Landa = landa M
asssociative 
distributive

'''


#----------
#Matrix multiplication
#a matrix Q --> size m x n and matrix R n x q 
#and so we have rsult P with m x q elements 
#and we say dot
A = np.array([
    [10,20,30],
    [40,50,60]
])

D = np.array([
        [ 2,  3,  5,  7],
        [11, 13, 17, 19],
        [23, 29, 31, 37]
    ])
E = A.dot(D)
E

'''

array([[ 930, 1160, 1320, 1560],
       [2010, 2510, 2910, 3450]])

'''

'''
Matrix multiplication is not ommutative QR /= RQ
is associative = Q(RES) = (QR)S

is also distribute --> (Q+R)S = QS + RS


also 
MI = IM = M


* ---> it do the elemntwise multiplication
@ --> for this operation

'''




#------
#matrix transpose --> AT --> ith row in Mt is equal to ith olumn in M
A.T
#(q + r)T = q T + r T
#(Q.R)T = r T . q T


#------symmetric matrix M --> Mt = M




#-----plotting matrix
# we cna have rectangular and geometric

#addition --> like translation
#scalar multipoication --> scaling
#projection on one axis,or on angle30
#rotation --> V [cos (30 sin(30) cos (120 sin(120]
#Other linear transformation
#u= [x,y,z] 
#F= [a b c , d e f]
#F(U)


#-----iNVERSE
LA.inv(F_shear)
#-1
#m.m-1 = M-1.M=I

#(landa x M)-1 = 1/landa * M-1



#----Orthogonal matrix
#H-1 = HT

#determinant
#|M\ = M1,1 x |m(1,1) .....
M = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 0]
    ])
LA.det(M)
#27








