import matplotlib.pyplot as plt
import numpy as np

def p(x):
    return 1+x**1

def q(x):
    return 1-x

def f_0(x):
    return 6*x+p(x)*(3*x**2)+q(x)*(x**3+2) 

def f(x,u,v):
    return v
def g(x,u,v):
    return f_0(x) - p(x) * v - q(x) * u

a, b = 1, 2
beta = 1.0
y_a, y_b = 6, 10

N=1000000
h = (b-a)/(N-1)

A = [[]]
for i in range(0,N):
    A.append([])

A[0].append(0)
A[0].append(1-beta/h)
A[0].append(beta/h)

A[N-1].append(0)
A[N-1].append(1)
A[N-1].append(0)

def x_(i):
    return a + h * i

for i in range(1,N-1):
    A[i].append(1/h**2-p(x_(i))/2/h)
    A[i].append(-2/h**2+q(x_(i)))
    A[i].append(1/h**2+p(x_(i))/2/h)

F=[]
F.append(y_a)
for i in range (1,N-1):
    F.append(f_0(x_(i)))
F.append(y_b)

alpha = []
beta = []

alpha.append(-A[0][2]/A[0][1])
beta.append(-F[0]/A[0][1])

for i in range (1,N):
    alpha.append(-A[i-1][2]/(A[i-1][0]*alpha[i-1] + A[i-1][1]))
    beta.append((F[i-1]-A[i-1][0]*beta[i-1])/(A[i-1][0]*alpha[i-1] + A[i-1][1]))

x=[]
u=[]
for i in range (0,N):
    x.append(x_(i))
    u.append(0)

u[N-1] = (F[N-1]-A[N-1][0]*beta[N-1])/(A[N-1][1] + A[N-1][0] * alpha[N-1])

for i in range (N-2,-1,-1):
    u[i] = alpha[i+1] * u[i+1] + beta[i+1]

plt.plot(x,u,color='b')

x_ = np.array(x)
plt.plot(x_,x_**3+2,color='r')

plt.xlim(left=a,right=b)
#plt.show()

occuracy = 0
for i in range(0,len(x)):
    if abs(u[i] - (x[i]**3+2)) > occuracy:
        occuracy = abs(u[i] - (x[i]**3+2))

with open("data.txt", "a") as myfile:
    myfile.write(str(h) + "\t" + str(occuracy)+'\n')