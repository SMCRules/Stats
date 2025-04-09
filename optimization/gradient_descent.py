"""
The steps for the gradient descent algorithm are given below:

1. Choose a random initial point x_initial and set x[0] = x_initial

2. For iterations t=1..T
    Update x[t] = x[t-1] - eta * ∇f(x[t-1])
Gradient descent and steepest descent are the same algorithm whenever
for the direction of motion dk, is chosen as -∇f(xk). 

"""