
import random
import math

"""
Buffon's needle estimate of pi
"""

N = int(input("Enter number of simulations: "))
d = 5
l = 3 # We have to set l that satisfy l<=d.
i = 0
S = 0
while i < N:
    X = random.uniform(0, 1)
    Z = random.uniform(0, 1)
    r = math.sqrt(X**2 + Z**2)
    if r < 1:
        sine_theta = Z / r
        i += 1
        z = random.uniform(0, d/2)
        if z <= l / 2 * sine_theta:
            S += 1

print(2 * l * N / (S * d))