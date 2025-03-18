'''

In The Name Of GOD

Ali Pilehvar Meibody


Table of Contents:

Numerical Modeling



'''

#=======================================================
#=======================================================
#=======================================================
#=======================================================
#-----------------------Differentiation-----------------
#=======================================================
#=======================================================
#=======================================================
#=======================================================


import sympy as sp
# Define the symbol
x = sp.symbols('x')
# Define the function
f = sp.sin(x) * sp.exp(x)
# Compute the derivative
dfdx = sp.diff(f, x)
print(f"Derivative: {dfdx}")


#also higehr order differentiation
d2fdx2 = sp.diff(f, x, 2)  # Second derivative
print(d2fdx2)


#at specific point
value = dfdx.subs(x, 2)  # Evaluate at x = 2
print(value)


#consider we have y
diff(y**2, x)

#higher order
diff(y**2, x, x)
#or
diff(y**2, x, 2) # same as above




#To calculate the derivative of a multivariate expression, we can do:-----
x, y, z = symbols("x,y,z")
f = sin(x*y) + cos(y*z)
#d3f / dx d2y
diff(f, x, 1, y, 2)






#_--ALSO YOU Can go for numerical----
#when the symbolic is not available or you want to go for numerical

import numpy as np
from scipy.misc import derivative

# Define the function
def f(x):
    return np.sin(x) * np.exp(x)

# Compute the derivative at x = 2
dfdx_numeric = derivative(f, 2.0, dx=1e-5)
print(f"Numerical Derivative at x=2: {dfdx_numeric}")


#he dx parameter defines the step size for the finite difference approximation.
#This method is useful when you don’t have a symbolic representation of the function.



#----with numpoy if you have points not the functions
import numpy as np
# Define discrete x values
x = np.linspace(0, 10, 100)
y = np.sin(x) * np.exp(x)  # Function values
# Compute numerical derivative
dy_dx = np.gradient(y, x)
print(dy_dx[:5])  # Print first 5 derivatives
#his method is useful when you have discrete data points instead of an analytical function.









#=======================================================
#=======================================================
#=======================================================
#=======================================================
#-----------------------Integration---------------------
#=======================================================
#=======================================================
#=======================================================
#=======================================================

#f
integrate(f, x)

#also you can specify the limti 
integrate(f, (x, -1, 1))

integrate(exp(-x**2), (x, -oo, oo))

#oo for sympy means infinity



#----Numerical integration: quadrature----
'''
is called numerical quadrature, or simply quadature. SciPy provides a series
of functions for different kind of quadrature, for example the quad, dblquad
and tplquad for single, double and triple integrals, respectively.

'''
from scipy.integrate import quad, dblquad, tplquad
# define a simple function for the integrand
def f(x):
    return x


x_lower = 0 # the lower limit of x
x_upper = 1 # the upper limit of x

val, abserr = quad(f, x_lower, x_upper)

print("integral value =", val, ", absolute error =", abserr) 
#	integral value = 0.5 , absolute error = 5.55111512313e-15



#also we have args-------
def integrand(x, n):
    """
    Bessel function of first kind and order n. 
    """
    return jn(n, x)

x_lower = 0  # the lower limit of x
x_upper = 10 # the upper limit of x

val, abserr = quad(integrand, x_lower, x_upper, args=(3,))
print(val, abserr) 
#0.736675137081 9.3891268825e-13




#for mor simple fucntion we can use lambda
val, abserr = quad(lambda x: exp(-x ** 2), -Inf, Inf)

print("numerical  =", val, abserr)

analytical = sqrt(pi)
print("analytical =", analytical)
#numerical  = 1.77245385091 1.42026367809e-08
#analytical = 1.77245385091

#for scipy inf -inf
#for sympy --> 00 -00




#High dimentional integration----
def integrand(x, y):
    return exp(-x**2-y**2)

x_lower = 0  
x_upper = 10
y_lower = 0
y_upper = 10

val, abserr = dblquad(integrand, x_lower, x_upper, lambda x : y_lower, lambda x: y_upper)

print(val, abserr) 














'''
---------------------------------------
------- Numerical Moddeling -----------
---------------------------------------


We have these types of equations that must be solved:
In all point of science from heat transfer, mass transfer, fluid mechanic, 
rheology , molecular dynamic, quantum physic and … we have a lot of equations 
that can have solution that is root or maybe it is some equations that show 
the current state of system and needs to be solved. What is these equations? 
It can be something very lienar that onl;y when yousolve you get the exact 
like we have 20 distance and we are in x=0 and what is new x . some times 
is more complicated or non lienar or differential It means that it depends
on something else and changed based on that .

For solving the equation or equations in any forms we have these two important methods:


Analytical Methods provide exact solutions 
Numerical Methods, on the other hand, offer approximate solutions, which are 
especially useful when analytical solutions are difficult or impossible to obtain.


Process some operations is done on materials. It can be simple transferring from one
place to other or complicated chemical reaction.

Mathematical modeling achieving equations that represent that process as some condition and input values


** equations can be algebraic equation, differential and mix of them



Modeling approaches:
There is 4 approaches from theory (base), empirical (experimental), semi empirical , likelihood 

1-Theory (base)-> here we use one or multiple Conservation laws.  
it can be conservation laws of mass, energy, momentum or population.
Conservation laws format:

Aggregation = input – output + production – consumption

Then we must select variables.

Dependent variable and independent variable.
Actually in most cases we have dependent variable (Y) that depend on one independent variable (X) and we can change the independent variable and we want to calculate or se the dependent variable (Y).

We write the conservation laws for element region.
Element is the biggest region that dependent variable (y) is consistent. This element can be system or control volume. 


System (sys)  mass is constant. From borders it can transfer Energy and momentum.
Control volume (cv) has open borders it can transfer Mass, Energy and momentum.

For writing the conservation laws we have two aspects integral (Lumped) and differential(distributive)

In integral (lumped) method, averages of parameter in element is considered and we ignore distribution of that parameter on place.
In differential (distributive) method, we can not ignore the distribution and we can not use average of parameter in element and we must consider the distribution of that on place.

2- empirical (Experimental) approach:
We used experiments to measure some variables  fit curving

3-semi empirical (semi experimental) approach:
With considering theory we use one model and then we achieve the models parameter with data and curve fitting.

4-likehood two different plan that has same mechanism


#-----------
Modeling Steps:

In modeling we have two phase first phase is model development and second phase is model solution (simulation).

1-Goal definition  determination of properties and numerical variables, determination of dependent variable and independent variable, determination of accuracy

2-data preparing  process diagram, assumptions, data , element selection

3-formulation conservation law, constitutive equations, governing equations

……….

4-Determination of solution: analytical , numerical , graphical approach

5-analysis of results  analysis sensitivity and interaction and relationship between variables

6-model validity (validation)  comparison with experimental results or more complicated models.


In summary  first determination of goal and variables, element region, conservation laws and numerical, analytical and then analysis and comparison.




Flashback 
In integral (lumped) method, averages of parameter in element is considered and we ignore distribution
of that parameter on place.

In differential (distributive) method, we can not ignore the distribution and we can not use average of
parameter in element and we must consider the distribution of that on place.



''''





#===========================================
'''   1- Linear Algebric Equations      '''
#===========================================
#something like ax+b=0

'''
Analytical
'''
#solution--> analythical ax=-b , x=-b/a
from sympy import symbols , solve

x= symbols('x')
eq=8*x + 4
sol = solve(eq,x)
print(sol) #-1/2


#some notes on sympy
#n SymPy we need to create symbols for the variables we want to work with. We can create a new symbol using the Symbol class
x = Symbol('x')
# alternative way of defining symbols
a, b, c = symbols("a, b, c")

#or also assumption
x = Symbol('x', real=True)
x.is_imaginary
x = Symbol('x', positive=True)
1+1*I

#There are three different numerical types in SymPy: Real, Rational, Integer:
r1 = Rational(4,5)


#we have also Numerical evaluation
pi.evalf(n=50)

#or
y = (x + pi)**2
N(y, 5) # same as evalf


#Use the function lambdify to "compile" a Sympy expression into a function that is much more efficient to evaluate numerically
f = lambdify([x], (x + pi)**2, 'numpy')  # the first argument is a list of variables that
                                         # f will be a function of: in this case only x -> f(x)
y_vec = f(x_vec)  # now we can directly pass a numpy array and f(x) is efficiently evaluated



#Expand ------
expand((x+1)*(x+2)*(x+3))
#x**3 + 3**x*2 + .....



sin(a+b)
expand(sin(a+b), trig=True)



#Simplify-----
simplify((x+1)*(x+2)*(x+3))

simplify(sin(a)**2 + cos(a)**2)

simplify(cos(x)/sin(x))

#aprt and together
f1 = 1/((a+1)*(a+2))
apart(f1)

f2 = 1/(a+2) + 1/(a+3)
together(f2)


#Sums and products
n = Symbol("n")
#sigma
Sum(1/n**2, (n, 1, 10))

#or
Sum(1/n**2, (n,1, 10)).evalf()

#or
Sum(1/n**2, (n, 1, oo)).evalf()

#product
Product(n, (n, 1, 10)) # 10!


#limit---
limit(sin(x)/x, x, 0)


#series-----
#Series expansion is also one of the most useful features of a CAS
series(exp(x), x)
#1 + x + x**2 /2 + x**3/6 + .....
#by default it is around x=0

#if you want to change that
series(exp(x), x, 1)

#if you want to say untill which order
series(exp(x), x, 1, 10)

#also for approximation
s1 = cos(x).series(x, 0, 5)
s2 = sin(x).series(x, 0, 2)
expand(s1 * s2)



#also we have special from scipy
from scipy.special import jn, yn, jn_zeros, yn_zeros
n = 0    # order
x = 0.0
# Bessel function of first kind
print("J_%d(%f) = %f" % (n, x, jn(n, x)))
x = 1.0
# Bessel function of second kind
print("Y_%d(%f) = %f" % (n, x, yn(n, x)))





x = linspace(0, 10, 100)

fig, ax = plt.subplots()
for n in range(4):
    ax.plot(x, jn(n, x), label=r"$J_%d(x)$" % n)
ax.legend();




# zeros of Bessel functions
n = 0 # order
m = 4 # number of roots to compute
jn_zeros(n, m)



#it doesnt need numerical *** becauae it is easy and we can only use sympy 









#===========================================
'''   2- Non-Linear Algebric Equations   '''
#===========================================
'''
it can be quadtatic ax**2 + bx + c =0
cubic --> ax**3 + bx**2 + cx + d=0
exponential : e**x - 4 =0
trigonometric : sin(x) = 0l.5

'''


'''
------------Analytical---------
--> Factoring using algebric identities
x2-5x+6=0
(x-2)(x-3)=0

quadratic formula --> x = -b +- radical / 2**a


'''
#we can use sympy for this -->

from sympy import symbols , solve

x= symbols('x')
eq=x**2 - 8*x + 4
sol = solve(eq,x)
print(sol) #[4 - 2*sqrt(3), 2*sqrt(3) + 4]

x= symbols('x')
eq=x**2+1 
sol = solve(eq,x)
print(sol) #[-I, I]


x= symbols('x')
eq=x**2 - 4*x + 4
sol = solve(eq,x)
print(sol) #..........



x= symbols('x')
eq=x**3 - 8*x + 4
sol = solve(eq,x)
print(sol) #........


x = symbols('x')
eq = sin(x) - 0.5
sol = solve(eq, x)
print(sol)  # Uses trig simplifications internally



x, y = symbols('x y')
expr = log(x*y**2)
expanded = expand_log(expr, force=True)
print(expanded)  # Outputs: log(x) + 2*log(y)


'''
SYMPY


linsolve()	Analytical	Solves linear systems using matrices.
nonlinsolve()	Analytical	Solves nonlinear systems symbolically.
solve_linear_system()	Analytical	Solves a linear system in augmented matrix form.
solve_univariate_inequality()	Analytical	Solves inequalities for a single variable.
dsolve()	Analytical	Solves differential equations.
pdsolve()	Analytical	Solves partial differential equations.
diophantine()	Analytical	Solves Diophantine equations (integer solutions).



nsolve()	Numerical	Finds numerical solutions using Newton’s method.
'''



'''
------------Graphical---------
we can get in the f(x) =0 --> and draw that
and see at which poin we habve th cross teh y=0


'''




#-------NUMERICAL-------
#for moth non-linear equation we use nuemrical methods

'''
we have different techniques

iteration methods---> g(x) = x with intiial guess and .....

wegeshtain method ---> more speed up because we have x1=g(x0) so
x m+1 = xm-1 g(x) .....


Newton Method --> first we have initial guess (x0) and then we havederivatiev f'(x0) and 
we must know each cross y=0 at which point and this is our new x and .......
f'(x0) = f(x0) / x0 - x1
X m+1 = xm - f(xm)/f'(xm)

secant method --> instead of f'() we use the approixmated of that 


Bisection method --> f(a) and f(b) must be non same sign and ..
c= a+b/2 --> f(c) and then ....


'''


#---bisection method----
from scipy.optimize import bisect

def f(x):
    return x**3 - x - 2  # Example function

root = bisect(f, 1, 2)  # Bracketing between 1 and 2
print(root)
#slow



#---Newton’s Method------
from scipy.optimize import newton

def f(x):
    return x**3 - x - 2

def df(x):  # Derivative
    return 3*x**2 - 1

root = newton(f, x0=1.5, fprime=df)  # Starts at x0=1.5
print(root)

#if initila guess bad or thederivative go to zero is bad


#sympy is more for exact and analytival but it has nsolve for enwton 
#urpose: Solves nonlinear equations numerically using Newton’s method.
#🔹 Uses: Newton-Raphson method (iterative)
from sympy import symbols, Eq, nsolve

x = symbols('x')
eq = Eq(x**3 - x - 2, 0)

root = nsolve(eq, x, 1.5)  # Initial guess = 1.5
print(root)  # Output: 1.52137970680457

#also for system----
from sympy import symbols, nsolve

x, y = symbols('x y')

eq1 = x**2 + y**2 - 4
eq2 = x - y - 1

solution = nsolve((eq1, eq2), (x, y), (1, 1))  # Initial guesses
print(solution)  # Output: (1.622, 0.622)








#----alternatiev( Secant Method (Newton's Without Derivative))
root = newton(f, x0=1.5, x1=2.0)  # Uses two initial guesses
print(root)
##No derivative needed
# Slower than Newton’s method


#---Fixed-Point Iteration-----
from scipy.optimize import fixed_point

def g(x):
    return (x + 2/x)**0.5  # Example transformation

root = fixed_point(g, x0=1.5)
print(root)




#-----(ROOT) ---> GENERAL FOR ALL the equation or system of non-linear
#Uses various numerical solvers (Newton, Broyden, Hybr, LM, etc.).

from scipy.optimize import root

result = root(fun, x0, method='method_name', jac=jacobian)


from scipy.optimize import root

def f(x):
    return x**3 - x - 2

x0 = 1.5  # Initial guess

sol = root(f, x0, method='hybr')  # Hybrid method (default)
print(sol.x)  # Root found



'''

hybr	Trust Region	General Problems (Default)
lm (Levenberg-Marquardt)	Trust Region	Nonlinear Least Squares
broyden1	Quasi-Newton	Large Sparse Problems
broyden2	Quasi-Newton	Large Sparse Problems
anderson	Iterative	Fixed-Point Problems
krylov	Iterative	Large Systems
diagbroyden	Quasi-Newton	Diagonal Jacobians
excitingmixing	Iterative	Physics & Engineering
linearmixing	Iterative	Fixed-Point Equations





****
For a single nonlinear equation, use simpler functions like newton().
For systems of equations, root() is the best choice.

'''



















#===========================================
'''  3- Linear Algebric system Equations '''
#===========================================
'''
A system of linear equations consists of two or more linear equations with the same set of variables. The general form for two variables is:

Before that we must know some introduction of matrices
We have scaler and vector and tensors.
Matrices are arrays of numbers. It can helpo us to compact the equations and fast calculation.


analytically Substitution --> one to anotheer one
 Elimination --> minus between two things
 
Cramer’s rule  --> X = | | / | | --> not efficient
Gussian Elimination ---> Can have numerical instability
Gauss Jordan 
Matrix Inversion:--->Ax=B  A-1 Ax=A-1 b  Ix=x=A-1b
LU Decomposition
Cholesky Decomposition

'''

#Gaussian Elimination / Substitution
from sympy import symbols, Eq, solve
# Define variables
x, y = symbols('x y')
# Define equations
eq1 = Eq(2*x + 3*y, 7)
eq2 = Eq(4*x - y, 5)
# Solve the system
solution = solve((eq1, eq2), (x, y))
print(solution)  # {x: 2, y: 1}



#Matrix Row Reduction (RREF) , like Gussian but in matrix form
#or if you have matrix based-------
from sympy import Matrix, symbols, linsolve
# Define variables
x, y = symbols('x y')
# Define coefficient matrix and right-hand side
A = Matrix([[2, 3], [4, -1]])  # Coefficients
b = Matrix([7, 5])  # Right-hand side
# Solve system
solution = linsolve((A, b), x, y)
print(solution)  # {(2, 1)}




#for augmented matrices------
from sympy import solve_linear_system
# Augmented matrix
system = Matrix([[2, 3, 7], [4, -1, 5]])
# Solve system
solution = solve_linear_system(system, x, y)
print(solution)  # {x: 2, y: 1}


#***For large systems, linsolve() is recommended because it is optimized for matrix computations.


#Only gussian elimination
from sympy import Matrix
# Augmented matrix [A|B]
system = Matrix([[2, 3, 7], [4, -1, 5]])
# Perform row reduction
system_rref = system.rref()  # Reduced Row Echelon Form
print(system_rref)  # (Matrix([[1, 0, 2], [0, 1, 1]]), (0, 1))


#LU Decomposition (Lower-Upper Factorization)
# Use Case: Efficient for large systems and iterative methods.
from sympy import lu
# LU Decomposition
L, U, _ = A.LUdecomposition()
# Solve LY = B (Forward substitution)
Y = L.solve(B)
# Solve UX = Y (Backward substitution)
X = U.solve(Y)
print(X)  # Matrix([[2], [1]])


#Cholesky Decomposition (For Symmetric Positive Definite Matrices)
# Use Case: Only for symmetric, positive-definite matrices.
from sympy import cholesky
# Define a symmetric positive definite matrix
A_cholesky = Matrix([[4, 2], [2, 3]])
# Compute Cholesky decomposition (A = LL^T)
L = cholesky(A_cholesky)
print(L)  # Lower triangular matrix




#matrix inversion
#Use Case: Only works when A is invertible (det(A) ≠ 0).
#Expensive for large systems
A_inv = A.inv()  # Compute inverse
X = A_inv * B  # Compute solution
print(X)  # Matrix([[2], [1]])







#-------We can use LU decomposition for linear equations with SCIPY
#here we dont have the equation sysmbolic so we must sue something else like 
#coefficient matrix from numpy array

#-------Analytical (LU Decomposition methods)------------ 
#useful for large-scale or ill-conditioned systems 

from scipy.linalg import solve

A = np.array([[3, 2], [1, 4]])
b = np.array([5, 6])

x = solve(A, b)  # Gaussian Elimination internally
print(x)



#For large or ill-conditioned systems, iterative numerical methods are better.------

'''
Jacobi Method
Iteratively updates values based on the previous iteration.
Good for diagonally dominant matrices.


Gauss-Seidel Method
Faster than Jacobi because it updates values immediately instead of waiting for the next iteration.

Conjugate Gradient (CG)
Used for sparse and large systems.
Common in machine learning and physics simulations.


'''
#iterative method----Best for Large Sparse Systems
from scipy.sparse.linalg import cg

A = np.array([[4, 1], [1, 3]])
b = np.array([1, 2])

x, info = cg(A, b)
print(x)

#Only works for symmetric positive definite matrices




#Generalized Minimal Residual Method (GMRES)
from scipy.sparse.linalg import gmres

A = np.array([[3, 2], [1, 4]])
b = np.array([5, 6])

x, info = gmres(A, b)  # Iterative GMRES solver
print(x)
#Can be slow if the system is not well-conditioned








#===========================================
''' 4-Non-Linear Algebric system Equations '''
#===========================================


#Substitution
#but fail for highly non linear

from sympy import symbols, Eq

x, y = symbols('x y')
eq1 = Eq(x**2 + y**2, 4)
eq2 = Eq(x + y, 2)

solutions = solve((eq1, eq2), (x, y))
print(solutions)



#Uses Gröbner basis, algebraic manipulation, and radicals.
from sympy import nonlinsolve
solutions = nonlinsolve([x**2 + y**2 - 4, x**3 - y - 1], (x, y))
print(solutions)


#recommended
#reduce_system() (Simplifies Before Solving)
from sympy.solvers.solveset import reduce_system
reduced_system = reduce_system([x**2 + y**2 - 4, x + y - 2], (x, y))
print(reduced_system)



#-----(ROOT) ---> GENERAL FOR ALL the equation or system of non-linear
#Uses various numerical solvers (Newton, Broyden, Hybr, LM, etc.).

from scipy.optimize import root

result = root(fun, x0, method='method_name', jac=jacobian)


from scipy.optimize import root

def f(x):
    return x**3 - x - 2

x0 = 1.5  # Initial guess

sol = root(f, x0, method='hybr')  # Hybrid method (default)
print(sol.x)  # Root found



'''

hybr	Trust Region	General Problems (Default)
lm (Levenberg-Marquardt)	Trust Region	Nonlinear Least Squares
broyden1	Quasi-Newton	Large Sparse Problems
broyden2	Quasi-Newton	Large Sparse Problems
anderson	Iterative	Fixed-Point Problems
krylov	Iterative	Large Systems
diagbroyden	Quasi-Newton	Diagonal Jacobians
excitingmixing	Iterative	Physics & Engineering
linearmixing	Iterative	Fixed-Point Equations





****
For a single nonlinear equation, use simpler functions like newton().
For systems of equations, root() is the best choice.

'''









#===========================================
''' 5-Differential Equations '''
#===========================================

'''

Differential equations are mathematical equations that relate a function to its derivatives.
They play a critical role in modeling various physical, biological, and engineering systems,
as they describe how quantities change with respect to one another. Understanding the types
of differential equations and their applications is fundamental for solving real-world problems
in fields like fluid mechanics, heat transfer, and molecular dynamics.



Differential equations are fundamental in modeling various physical phenomena in engineering, physics, biology,
economics, and more. They describe the relationships involving rates of change and are classified into ordinary
differential equations (ODEs) and partial differential equations (PDEs). Solving these equations can be approached 
through two primary methods: analytical methods and numerical methods.
•	Analytical Methods provide exact solutions to differential equations, yielding formulas that describe 
the behavior of the system.
•	Numerical Methods, on the other hand, offer approximate solutions, which are especially useful when
analytical solutions are difficult or impossible to obtain.



ORDINARY DIFFERENTIAL EQUATION (ODE)


F(x,y,y',y'',...)=0

first order --> dy/dx + y =0 
second order --> d2y/d2x + dy/dx + 2y=0



PARTIAL DIFFERENTIAL EQUATION (PDE)
Multiple independent variables and their partial derivative

F(x1,x2,y, dy/dx1 , dy/dx2 ,...) =0

First Order --> ro u / ro x + c * ro u / ro t =0
Second Order --> ro2 u / ro2 x + c * ro2 u / ro2 t =0


'''



#===============================================
#===============================================

'''                  ODE                      '''
#===============================================
#===============================================

#======================================
#-----ANALYTHICALLY--------------
'''
analytical way for ODE Equatio


First order
seperation of variables --> dy/dx = g(x)h(y)
linear Integrating factor ---> dy/dx + P(x)y = Q(x)
Exact --> M(x,y)dx + N(x,y)dy =0 --> check if roM/roy = roN/ rox
Bernulli --> dy/dx + P(x)y = Q(x)y**n , substition v=y**1-n




second order
Homogenious y'' + py' + qy=0
Non Homogenious y'' + p(x)y' + q(x)y= f(x)

Higher-Order ODEs
Solved using reduction of order, Laplace transforms, or series solutions.

for isntance
Homogeneous Linear Higher-Order ODEs
we can assume the answer is y=e**x
and finally we have different r 
y1=C1 + C2 + ....

Non-homogenious--> work with simpler F(x) or variation in parameters




In all of them we must analysis all of them and then
we can go for numericals


sympy for analytical


Analytical methods give exact solutions but are limited to simpler equations.

'''


#---Separation of Variables:-----
#-----Integrating Factor Method --> DY/DX + p(x)y = q(x)
#Homogeneous and Non-Homogeneous Linear Equations:---
#----y'' + p(x)y' + q(x)y=0
#-----y'' + p(x)y' + q(x)y= f(x)
#Series Solutions (Power Series):
#First-Order ODE: Separable Equation
#dy/dx=xy
from sympy import symbols, Function, Eq, dsolve
x = symbols('x')
y = Function('y')(x)
ode = Eq(y.diff(x), x * y)
solution = dsolve(ode, y)
print(solution)



#Linear First-Order ODE
#dy/dx + y =x
ode = Eq(y.diff(x) + y, x)
solution = dsolve(ode, y)
print(solution)


#Second-Order ODE (Homogeneous)
#y'' + 'y + 2y=0
x = symbols('x')
y = Function('y')(x)
ode = Eq(y.diff(x, x) + 3*y.diff(x) + 2*y, 0)
solution = dsolve(ode, y)
print(solution)


# Non-Homogeneous ODE
#y'' + y' = sinx
ode = Eq(y.diff(x, x) + y, symbols('sin')(x))
solution = dsolve(ode, y)
print(solution)






#======================================
#-----Numerical--------------
'''
Euler’s Method (First-Order ODEs)

Basic idea: Approximate y(x) using small steps h


'''
import numpy as np
import matplotlib.pyplot as plt
def euler(f, x0, y0, x_end, h):
    x = np.arange(x0, x_end + h, h)
    y = np.zeros(len(x))
    y[0] = y0
    for i in range(1, len(x)):
        y[i] = y[i-1] + h * f(x[i-1], y[i-1])
    return x, y
# Define ODE: y' = y - x^2 + 1
f = lambda x, y: y - x**2 + 1
# Solve using Euler's method
x, y = euler(f, 0, 0.5, 2, 0.1)
# Plot results
plt.plot(x, y, 'bo-', label="Euler Approximation")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()



'''
Improved Euler (Heun’s Method)
More accurate than Euler. / Uses the average slope:
'''


'''
Runge-Kutta Methods (RK)
Most commonly used!
Fourth-order Runge-Kutta (RK4):
'''
from scipy.integrate import solve_ivp

def ode_system(t, y):
    return y - t**2 + 1

sol = solve_ivp(ode_system, (0, 2), [0.5], method='RK45', t_eval=np.linspace(0, 2, 20))

plt.plot(sol.t, sol.y[0], 'ro-', label="RK4 Approximation")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()



#----------------------------------------
#Solving Higher-Order ODEs Numerically

#Convert to a System of First-Order ODEs
#y'' + y =0.  --> define y1=y , y2=y' . y1'=y2.  y2'=-y1
def system(t, Y):
    y1, y2 = Y
    return [y2, -y1]  # dy1/dt = y2, dy2/dt = -y1

t_eval = np.linspace(0, 10, 100)
sol = solve_ivp(system, (0, 10), [1, 0], t_eval=t_eval, method='RK45')

plt.plot(sol.t, sol.y[0], label="y(t)")
plt.plot(sol.t, sol.y[1], label="y'(t)")
plt.legend()
plt.show()


#Boundary Value Problems (BVPs)
#y'' +y =0 with y(0) , y(pi/2)=1

from scipy.integrate import solve_bvp

def bvp_func(x, y):
    return np.vstack([y[1], -y[0]])

def boundary(ya, yb):
    return np.array([ya[0], yb[0] - 1])

x = np.linspace(0, np.pi/2, 10)
y_init = np.zeros((2, x.size))
sol = solve_bvp(bvp_func, boundary, x, y_init)

plt.plot(sol.x, sol.y[0], 'b-', label="BVP Solution")
plt.legend()
plt.show()



#non linear ODE-------


'''
SOLVING SYSTEM OF ODE



ANALYTICAL--------
(A) Direct Integration (Separable Equations)
(B) Matrix Exponential Method (For Linear Systems)
(C) Laplace Transform: 
Convert the system into algebraic equations, solve, and take the inverse Laplace transform.
Useful for linear systems with constant coefficients.
Laplace Transform (only for linear systems)

'''
import sympy as sp

t = sp.Symbol('t')
X = sp.Matrix(sp.symbols('x y')).T
A = sp.Matrix([[2, 1], [3, 2]])

# Compute the matrix exponential e^(At)
sol = sp.exp(A*t) * sp.Matrix([1, 0])  # Assuming initial condition X(0) = [1,0]
sol.simplify()
print(sol)



#------
s = sp.Symbol('s')
X_s = sp.Matrix(sp.symbols('X Y')).T
eqs = (s*X_s - sp.Matrix([1, 0])) - A * X_s
sol_s = sp.solve(eqs, X_s)  # Solve in Laplace domain
sol_t = [sp.inverse_laplace_transform(sol, s, t) for sol in sol_s.values()]
print(sol_t)


'''
Numerical Methods
'''

'''
(A) Euler’s Method (First-Order Approximation)
LIKE ONE ODE
'''
import numpy as np
import matplotlib.pyplot as plt

def f(t, X):
    x, y = X
    return np.array([2*x + y, 3*x + 2*y])  # dx/dt, dy/dt

t0, tf, h = 0, 2, 0.1
t_values = np.arange(t0, tf+h, h)
X_values = np.zeros((len(t_values), 2))
X_values[0] = [1, 0]  # Initial condition

for i in range(1, len(t_values)):
    X_values[i] = X_values[i-1] + h * f(t_values[i-1], X_values[i-1])

plt.plot(t_values, X_values[:, 0], 'r-', label='x(t)')
plt.plot(t_values, X_values[:, 1], 'b-', label='y(t)')
plt.legend()
plt.show()



'''
B) (B) Runge-Kutta Methods (RK4)
most common like ode
'''
from scipy.integrate import solve_ivp

def system(t, X):
    x, y = X
    return [2*x + y, 3*x + 2*y]

sol = solve_ivp(system, (0, 2), [1, 0], t_eval=np.linspace(0, 2, 50), method='RK45')

plt.plot(sol.t, sol.y[0], 'r-', label="x(t)")
plt.plot(sol.t, sol.y[1], 'b-', label="y(t)")
plt.legend()
plt.show()



'''
C) (C) Implicit Methods (Backward Euler, BDF)
'''

sol = solve_ivp(system, (0, 2), [1, 0], method='BDF')

plt.plot(sol.t, sol


         



#===============================================
#===============================================

'''                  PDE                      '''
#===============================================
#===============================================



#Solving PDEs Analytically (Wave Equation)
from sympy import Function, Derivative
t, x, c = symbols('t x c')
u = Function('u')(x, t)
wave_eq = Eq(Derivative(u, t, t), c**2 * Derivative(u, x, x))
print(wave_eq)



#---------------------------------------------
'''
(A) Separation of Variables
Used for linear PDEs where variables can be separated.
Converts a PDE into ordinary differential equations (ODEs).


'''
import sympy as sp

x, t = sp.symbols('x t')
X = sp.Function('X')(x)
T = sp.Function('T')(t)
alpha, lambd = sp.symbols('alpha lambda')

eq1 = sp.Eq(T.diff(t), -lambd * alpha * T)  # Time equation
eq2 = sp.Eq(X.diff(x, 2), -lambd * X)      # Space equation

sol_T = sp.dsolve(eq1, T)  # Solve ODE in time
sol_X = sp.dsolve(eq2, X)  # Solve ODE in space

print(sol_T, sol_X)



'''
(B) Fourier Series & Transform Methods
'''
from sympy.integrals.transforms import fourier_transform

u = sp.Function('u')(x)
F = fourier_transform(u, x, sp.Symbol('k'))
print(F)  # Fourier transform of u(x)


'''
(C) Laplace Transform
Used for initial/boundary value problems.
Converts PDEs into algebraic equations in the Laplace domain.

'''
from sympy.integrals.transforms import laplace_transform

f = sp.Function('f')(t)
F = laplace_transform(f, t, sp.Symbol('s'))  # Laplace transform of f(t)
print(F)



#=========================
#=========================
#------NUMERICAL-----------

#(A) Finite Difference Method (FDM)
import numpy as np
import matplotlib.pyplot as plt

# Parameters
L, T = 1.0, 0.2  # Length and time
Nx, Nt = 20, 100  # Grid points
dx, dt = L/Nx, T/Nt
alpha = 0.01  # Diffusion coefficient
r = alpha * dt / dx**2  # Stability condition

# Initialize solution
u = np.zeros((Nx+1, Nt+1))
x = np.linspace(0, L, Nx+1)
u[:, 0] = np.sin(np.pi * x)  # Initial condition

# Time-stepping loop
for n in range(0, Nt):
    for i in range(1, Nx):
        u[i, n+1] = u[i, n] + r * (u[i+1, n] - 2*u[i, n] + u[i-1, n])

# Plot result
plt.imshow(u, extent=[0, T, 0, L], aspect='auto', origin='lower', cmap='hot')
plt.colorbar(label='Temperature')
plt.xlabel('Time')
plt.ylabel('Position')
plt.show()




#(B) Finite Element Method (FEM)
#Divides the domain into small elements.
#Used for complex geometrie


from fenics import *

# Define domain and mesh
mesh = UnitIntervalMesh(10)
V = FunctionSpace(mesh, "P", 1)

# Define boundary conditions
u_D = Expression("sin(pi*x[0])", degree=2)
bc = DirichletBC(V, u_D, "on_boundary")

# Define problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0)
a = dot(grad(u), grad(v)) * dx
L = f * v * dx

# Solve
u = Function(V)
solve(a == L, u, bc)

# Plot solution
import matplotlib.pyplot as plt
plot(u)
plt.show()





#(C) Finite Volume Method (FVM)
#Used in computational fluid dynamics (CFD).
## Install OpenFOAM and run CFD simulations


#(D) Method of Lines (MOL)
#Converts PDEs into ODEs and solves using solve_ivp().
from scipy.integrate import solve_ivp

def heat_eq(t, u, alpha, dx):
    dudx2 = (np.roll(u, -1) - 2*u + np.roll(u, 1)) / dx**2
    dudx2[0] = dudx2[-1] = 0  # Boundary conditions
    return alpha * dudx2

Nx, L, alpha = 20, 1, 0.01
x = np.linspace(0, L, Nx)
u0 = np.sin(np.pi * x)

sol = solve_ivp(heat_eq, (0, 0.2), u0, args=(alpha, x[1] - x[0]))

plt.imshow(sol.y, aspect='auto', origin='lower', cmap='hot')
plt.colorbar(label='Temperature')
plt.xlabel('Time Steps')
plt.ylabel('Position')
plt.show()












'''
drawback of analytical solution



 Limited to Simple Equations

 Difficulty with Nonlinear Equations

 Computational Complexity

 
Use of Special Functions: Some analytical solutions may require special functions 
(e.g., Bessel functions, Gamma functions, or Airy functions) that are not always 
easily understood or applicable to real-world systems. Additionally, the existence
of these special functions can make the solution less interpretable.





When to Use Numerical Methods Instead
Given these drawbacks, numerical methods are often preferred in the following situations:

When the equation is too complex or nonlinear for analytical methods.
When working with large systems of equations that cannot be easily solved analytically.
When approximations are acceptable, and you are interested in observing the behavior of the solution over time or across various conditions.
In real-world scenarios where data or initial conditions are noisy or uncertain.


'''


