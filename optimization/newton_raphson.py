"""
A Newton-Raphson application to the problem of finding the square root of a number b.
Finding the square root of b is equivalent to solving the equation f(x) = x^2 - b = 0. 
Newton-Raphson produces a sequence in x(n+1) 

x(n+1) = 1/2 * (x(j) + b/x(j)) 

until it stabilizes around a solution of f(x) = 0
"""
