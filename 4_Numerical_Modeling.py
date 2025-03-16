'''

In The Name Of GOD

Ali Pilehvar Meibody


Table of Contents:

Numerical Modeling



'''



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




#===========================================
''' 5-Differential Equations '''
#===========================================


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



#Solving PDEs Analytically (Wave Equation)
from sympy import Function, Derivative
t, x, c = symbols('t x c')
u = Function('u')(x, t)
wave_eq = Eq(Derivative(u, t, t), c**2 * Derivative(u, x, x))
print(wave_eq)



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

















