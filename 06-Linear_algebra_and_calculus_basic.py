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
Matrix multiplication is not commutative QR /= RQ
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




#================================
#================================
#================================
'''   Differential Calculus   '''
#================================
#================================
#================================

#https://github.com/ageron/handson-ml2/blob/master/math_differential_calculus.ipynb
'''
Calculus is the study of continuous change. It has two major subfields: differential calculus, which studies the rate 
of change of functions, and integral calculus, which studies the area under the curve. In this notebook, we will discuss
the former.



When we have one straight line , between any point of A and B if we have rise (vertical)
and run(hroizontal) --> 
Slope = rise / run = delta y / delta x = yb-ya / xb - xa
so we said slope.

What ablout CURVE?
for isntance we have y = f (x) = x**2
so we know that from left if we move to 0 and then to rigth, the slope is different.
so we don't have constant slope, but we have not constant slope, so for instance for 
point A , so we have point B from left to A and the slope is change
and when teh b is very close to A -> it is slope.
it means it is = xa


But note that --> Differentiability 
the function must be differentiable. for instance f(x) = |x| 
at x=0 --> in the left if B --> is -1 at that x=0 --> 0 , at rigth +
so we say at x=0 is not differentiable , but at other is differntiable.

'''

#---
'''
we have some constraints
1- functions must of course be defined as xa and so defined at xa=0
2-it must be cotinious
3-must not has breaking poiint
4-not vertical (liek x**3)

'''

#------
'''

f'(xa) = lim f(xb)- f(xa) / xb-xa   xb--->xa
so it is yb-ya/xb-xa and so
but actually if we get lim for instance for x**2 -->
we get 2*xa


we have tehse rules

lim c (x-->k) = c  constant
lim x = k  x-->k  if x approches
lim[f(x) + g(x) = limf(x) + lim g(x)
lim|f(x) X g(x) | = lim f(x) X lim g(x)


and also if we say that e = xa-b 
we can also  write 

f'(xa) = lim f(xa + e ) - f(xa) / e  when e-->0

'''


#-----god point
'''
f'=df/dx=d/dx f = Df = y.

f' --> lagrangfe notation
df/dx = leibniz notation
y. --> newtonian notation
Df --> Euler notation


'''


#-----Differentiation rules

'''
f(x) =c                 f'(x)=0
f(x)=g(x) + h(x)        f'(x) = g'(x) + h'(x)
f(x)=g(x)h(x)           f'(x)=g(x)h'(x) + h(x)g'(x)
f(x)=g(x)/h(x)          f'(x) = g'(x)h(x) - g(x)h'(x)/h**2(x)
f(x)=x**r               f'(x)=r*x**(r-1)
f(x)=ln(x)              f'(x)=1/x
f(x)=sin(x)             f'(x)=cos(x)
f(x)=cos(x)             f'(x)=-sin(x)
f(x)=tan(x)             f'(x)=1/cos(x)**2
f(x)=g(h(x))            f'(x)=g'(h(x))h'(x)


Also chain rule
df/dx = df/dy * dy/dx




'''


#========================================
#========================================
#Derivatives and optimization
#========================================
#========================================
'''
when we try to optimize function f(x) and we look for the values of x that
minimize (or maximize) the function.

so it means taht at minimum or maximu the function must be derivatiev and derivation must
be equal to zero. so one way is analytically find all values for which derivative is 0
and then determine which of tehse values optimzie the function.

Another option is perform Gradiant Descnt ( we will consider minimizng function , but process
for maximizing is sam). start with random point x0 and then use
function derivatiev to determien the slope at that point and move a little bit in the downward direction
then repeat the process until  you reach a local minimum and hopefully global minimum.

at each iteration, teh steo size is proportional to the slope, the process naturally slows down as it
approaches a local minimum. each step is also proportional to learning rate (hypeparameter)
There are many variants of Gradiant Descent algorithm. and they al rly on derivatiev of the cost
function with regards to the model paameetr.



'''

#---hIGHER ORDER DERIVATIVES
'''
FIR J=KNOWING THE DIFFERENTIATE THE FUCNTION F'(X) we have
f"(x) or d2f / dx2 also with repeating we have more than that. f"' or d3

'''

#---Partial Derivatives
'''
Up to now, we have only considered function with single variable (x). if
we have f(x,y) like sin(x,y) what happen?
so for inastance considder on surface and so at that point A 
if you want tostand at point A--> walk a;long x axis toward right(increasing x)( , your path 
woyld go 

so now ythe singl enumber is no longer sufficient to describe slope of the function at
that fiven point.
so , to find slope along x axis , we called partial derivatiof oif f with regard to x
ro f/ ro x 

ro f / ro x = lim (e-->0) f(x+e,y) - f(x,y) / e
so we consider uy constant.
also we have the partial regard to y.


so if at one point, all partial derivatives are defined and countious in neighhodd-> that fucntion
is totally differentiable at that point. soi we can locally aproximately by plane PA ( 
tangent plane to the surface at point A). 
z = f(xa,ya) + (x-xa)rof (xa,ya)/rox + (y-ya)rof(xa,ya)//roy
but in depplearning sometimes we can not go for thsat.


'''


#------Gradiants
'''
when we talk abotu single varibale function or two variables function
what abotu function with n variables
f(x1,x2,x3,....,xn) and so we say vector x -->
X=[x1 x2 x3 x4 5 .... xn]

now writng f(X) is easier than f(x1,x2,.....,xn)_
the gradient of the function f(x) at some point xa is the bvector whodse
compomnents are all the partial derivative of function at that point
delta f(xA) 

Delta f(Xa) = [ rof(XA)/ROX1   rof(xa)/rox2 ...... ro f(xa / roxn]

assuming the function is totally differentiable at point xa, the surface it describes casn be
approximated by plan at that point. so gradient vector is teh one that pojnts
towards teh steepest slope on tyat plane.

'''


'''

In deep learning --> gradient descent algorithm is absed on gradients, instead of derivatives
it work but using vectors ijnstead of scalars.


we start with vector x0 --> compute gradient of f at that point,. perfom small step
in opposite direction and repeate untillconvergence.

more precisely at each step t --> compute xt= xt-1 - neta deltaf(xt-1) 
so teh neta is learnign rate and very small. 

In deep learning x is input data which is feeded to neural network and get back a prediction
y^ = f(x) . we can say y^ = fw(x). w reprsent mdoel parameter.

so intraining, matrix X and vector y --> cinstan and w is bvariable --> try to minimize cost function
l x,y(w) = g(fx(w,y)-->g is function thayt measure discrepancy betwen presiction fx(w) and the labels y).
we minimize losss funcion with gradient descent (or variang of GD)--> we satrt with random model parameters w0 and
then we compute gradient  l(w0) --> and gradient decent step and repate to convrgence . 

'''


#-----------
'''
Jacobians

'''

'''
always we say that our output is single scalar, soemtimes it can be vector instead.
for exampel --> classification neural network typically putputs one probability for each class,
there are m classes, the neural network will output an d-dimensional vector for each input


In deep learning we generally only need to diferntiate teh loss function which almost always
outpus a singel scalar number. but suppose for secodn that you want to differntiate a function fz(x) 
which outptus d-dimensional vector.


The good news is that you can treat each output dimension independently of the others. This will give 
you a partial derivative for each input dimension and each output dimension. If you put them all in 
a single matrix, with one column per input dimension and one row per output dimension, you get the
so-called Jacobian matrix.




Jf (xa) =  [rof1(xA)/rox1 rof1(xA)/rox2 rof1(xA)/rox3  .... rof1(xA)/roxn ]
           [rof2(xA)/rox1 rof2(xA)/rox2 rof2(xA)/rox3  .... rof2(xA)/roxn ]
               ....           ....         .....               ......
            [rofn(xA)/rox1 rofn(xA)/rox2 rof1(xA)/rox3  .... ron(xA)/roxn ]

each row --> that fiunction (fxa) for all x input from x1 to xn at that point
and each columns --> different partial on x1,x2 and..

jacobian --> first order partial derivatiev of function f 


'''



#----------------
'''
Lets come back to fucntion f(x) with takes n-dimensioanl vector as input and outpus scaler
if you have partial derivatiev of f with regards to xi (the ith componen of x) and then
get new function X= ro f/roxi and then compute partial derivative of this fucntion with regars
to xj --> (the jth component of x) -->< we said partial derivatiev of partial derivative -->
scond order partial derivatives--> called Hessian.
X : ro**2f/ ro xj xi if i/= --> mixed second order partial derivative

f(x,y) = sin(x,y)

first order partial derivatives
rof/rox =y cos(xy)

ro f/roy =xcos(xy)



Now hessian ( derivatiev)

ro2f/rox2 = another derivatiev of first [second order] i=j
ro2f/roy2 = another derivatiev of first [second order] i=j


ro2f /rox roy= rof/rox [rof/roy] = rof/rox [xcos(xy)] = cos(xy) - xysin(xy)
ro2f /roy rox = rof/roy [rof/rox] = rof/roy [xcos(xy) = cos(xy) - xysin(xy)

so note that ro2/roxroy= ro2/royrox

The mateix containing al hesssiansis called Hessian matriix
Hf(xa) =  [ ro2f/rox1**2 (XA)  ro2f/rox1rox2 (XA) ..... ro2f/rox1roxn (XA)]
          [ ro2f/rox2**rox1 (XA)  ro2f/rox2**2 (XA) ..... ro2f/rox2roxn (XA)]
          [ ro2f/roxn**rox1 (XA)  ro2f/roxn**rox2 (XA) ..... ro2f/roxn**2 (XA)]
each row is second partial derivative of rof/rox to x1 only . but in each column the rof/rox change

There are greate optimization algorithem with take advantage of teh hessians but in
practuce deep learnign almost enevr uses them. indeed if funciton has n variable, there are
n*&*2 hyessians. since neural netwrok hass millions of parametrs --> hessians would exceed serveal billions of parametr -->
'''






