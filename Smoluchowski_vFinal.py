# This is my first time to simulate something related to mathematics in Python :|

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import quad
import time

start = time.time()

R   =   50       # The value of upper boundedness
Ih  =   100     # Number of x-nodes with integer indices (in space domain)
N   =   21     # Number of t-nodes (in time domain)
G   =   np.zeros((Ih,N+1))     # Initialize the approximation solution of g(t,x) as a grid of zeros
J   =   np.zeros((Ih+1,N))   # Initialize the approximation of the fluxes J as a grid of zeros
T   =   2        # The finished time T_end
Time=   np.linspace(0, T, N)

Y   =   np.linspace(0, R, Ih+1); # The array X with rational indices X[-1/2], X[1/2], X[3/2].... has the length Ih+1
X   =   np.zeros(Ih)  # The array X with integer index X[0], X[1],...., X[Ih-1],... has the length Ih

DeltaX      =   Y[1]-Y[0] # A uniform mesh in space domain
DeltaT      =   T/N
LambdaI     =   np.zeros((Ih,2))

g0_x   = lambda x: x * math.exp(-x)    # here I choose f0(x)=e^{-x}
a_dbx  = lambda x: 1/x    # Choose a(x,x') = 1 as Filbet's choice
logx   = lambda x: math.log(x, 10)

# Representation of the mesh: Equation (9) #
for i in range(Ih): # i is running from 0 to Ih-1
    X[i]           =   0.5*(Y[i]+Y[i+1]) # Representation of the meshes: Equation (9)
    LambdaI[i][0]  =   Y[i]    # Encode the lower limit of the interval Lambda[i]
    LambdaI[i][1]  =   Y[i+1]  # Encode the upper limit of the interval Lambda[i]
    G[i][0]        =   (1/DeltaX) * quad(g0_x, LambdaI[i][0], LambdaI[i][1])[0] # Pick the first value

# Initialize the value J[0][n]=0 as there is not any flux at the boundary?
for n in range(N):
    J[0][n] = 0

Alphaik = lambda i, k: i - k + 1 # Result of the uniform mesh

for n in range(N): # Fix a small number of loops to see the result
    print('Iteration', n, '-th over', N)
    for i in range(Ih):
        J[i+1][n] = 0
        I2        = np.zeros(i+1)
        SumI1     = np.zeros(i+1)
        for k in range(i+1):
            I2[k]    = quad(a_dbx, Y[i+1]-X[k], Y[Alphaik(i, k)])[0] * G[Alphaik(i, k)-1][n]
            for j in range(Alphaik(i, k), Ih): # Need to be revised
                SumI1[k] = SumI1[k] + G[j][n] * quad(a_dbx, LambdaI[j][0], LambdaI[j][1])[0]
            J[i+1][n] = J[i+1][n] + DeltaX * G[k][n] * (SumI1[k] + I2[k])
#    if n+1<N:
        G[i][n+1] = G[i][n] - (DeltaT/DeltaX) * (J[i+1][n] - J[i][n])

# Simulation for Equation (11) and (13) #
plt.plot(np.log(X), np.log(G[:, 2]), 'r-', label="t=0.20")
plt.plot(np.log(X), np.log(G[:, 6]), 'g-', label="t=0.60")
plt.plot(np.log(X), np.log(G[:, 10]), 'b-', label="t=1.00")
plt.plot(np.log(X), np.log(G[:, 15]), 'y-', label="t=1.50")
plt.xlabel('log(x)')
plt.ylabel('log(g(x))')
plt.title('Exact solution with kernel a(x,x_prime)=1 and f0(x)=exp(-x) in log scales')
plt.legend(loc='best')
plt.show()

# Configure the initial value g(0)=0 #
G_at_origin = np.zeros((1, N+1))
G           = np.row_stack((G_at_origin, G))
X           = np.append([0], X)
# Simulation for Equation (13)
plt.plot(X, G[:,2], 'r-', label="t=0.20")
plt.plot(X, G[:,6], 'g-', label="t=0.60")
plt.plot(X, G[:,10], 'b-', label="t=1.00")
plt.plot(X, G[:,15], 'y-', label="t=1.50")
plt.axis([0, 15, 0, 0.5])
plt.xlabel('x')
plt.ylabel('g(x)')
plt.title('Exact solution with kernel a(x,x_prime)=1 and f0(x)=exp(-x)')
plt.legend(loc='best')
plt.show()

# Simulation for Equation (10) #
plt.plot(X, G[:,0], label="t=0")
plt.axis([0, 15, 0, 0.5])
plt.xlabel('x')
plt.ylabel('g(x)')
plt.title('Initial approximation $g_i^0$')
plt.legend(loc='best')
plt.show()

# Simulation for Equation (12) #
plt.plot(X, J[:,2], 'r-', label="t=0.20")
plt.plot(X, J[:,6], 'g-', label="t=0.60")
plt.plot(X, J[:,10], 'b-', label="t=1.00")
plt.plot(X, J[:,15], 'y-', label="t=1.50")
plt.xlabel('x')
plt.ylabel('J')
plt.title('Fluxes J at kernel a(x,x_prime)=1 and f0(x)=exp(-x)')
plt.legend(loc='best')
plt.show()

end = time.time()
print(end-start)