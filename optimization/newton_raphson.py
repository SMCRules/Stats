"""
Newton-Raphson implementation of finding the square root of number b.
This is equivalent to solving the equation f(x) = x^2 - b = 0. 
Newton-Raphson produces a sequence in x(n+1) 

x(n+1) = 1/2 * (x(j) + b/x(j)) 

until it stabilizes around a solution of f(x) = 0
"""

import matplotlib.pyplot as plt

def newton_raphson_sequence(b=2, init=1.0, iterations=10):
    x = init
    sequence = [x]
    for _ in range(iterations):
        x = 0.5 * (x + b / x)
        sequence.append(x)
    
    return sequence

def f_evaluation(x, b):
    return x**2 - b

def plot_convergence(b=3, inits=[0.5, 2.0, 3.0, 4.0], iterations=10):
    plt.figure(figsize=(10, 6))
    
    for init in inits:
        x_seq = newton_raphson_sequence(b=b, init=init, iterations=iterations)
        f_vals = [f_evaluation(x, b) for x in x_seq]
        plt.plot(range(len(f_vals)), f_vals, marker='o', label=f'init = {init}')
    
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.title(f'Newton-Raphson Convergence for sqrt({b})')
    plt.xlabel('Iteration')
    plt.ylabel('f(x) = x² - b')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_convergence(b=3, inits=[0.5, 2.0, 3.0, 4.0], iterations=10)