'''
Reference: #https://github.com/ageron/handson-ml2/blob/master/math_linear_algebra.ipynb

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

x_coords, y_coords = zip(u, v)
plt.scatter(x_coords, y_coords, color=["r","b"])
plt.axis([0, 9, 0, 6])
plt.grid()
plt.show()


#ALSO WE CAN CREATE one vector for u , one for v


#3D VECTOR ----
#In space
a = np.array([1, 2, 8])
b = np.array([5, 6, 3]


from mpl_toolkits.mplot3d import Axes3D

subplot3d = plt.subplot(111, projection='3d')
x_coords, y_coords, z_coords = zip(a,b)
subplot3d.scatter(x_coords, y_coords, z_coords)
subplot3d.set_zlim3d([0, 9])
plt.show()



#---------------------
'''
Norm
the norm of vector (u) noted ||u|| is measure of length(magnitude) of u
we have a lot of norm but most common is Euclidian Norm

|| u|| = radical [  Zigma ui**2 ]
'''
def vector_norm(vector):
    squares = [element**2 for element in vector]
    return sum(squares)**0.5

print("||", u, "|| =")
vector_norm(u)

#or
$linalg --> linear algebra

import numpy.linalg as LA
LA.norm(u)










#differential_calculus
#https://github.com/ageron/handson-ml2/blob/master/math_differential_calculus.ipynb


