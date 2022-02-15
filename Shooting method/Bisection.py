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

def v_a(u_a):
    return (y_a - u_a)/beta
def psi(u,v):
    return u - y_b

N=10
h = (b-a)/N


def psi_wave(u_a,v_a,save,n,x_arr,res_arr):
    x = a
    u = u_a
    v = v_a
    while(x<b):
        if(save):
            x_arr[n].append(x)
            res_arr[n].append(u)
        du = f(x,u,v) * h
        dv = g(x,u,v) * h

        u += du
        v += dv
        x += h
    
    if(save):
        x_arr[n].append(x)
        res_arr[n].append(u)
    return psi(u,v)

x=[[]]
res=[[]]

left = -10
right = 10
eps = .001
err= 1

psi_left = psi_wave(left, v_a(left),False,0,x,res)
psi_right = psi_wave(right, v_a(right),False,0,x,res)
if not((psi_left < 0 and psi_right > 0) or (psi_left > 0 and psi_right < 0)):
    print("wrong area of searching")

D = (left + right)/2

step = 0

while(abs(err) > eps):
    psi_left = psi_wave(left, v_a(left),False,0,x,res)
    D = (left + right)/2

    x.append([])
    res.append([])
    psi_D = psi_wave(D, v_a(D),True,step,x,res)

    if (psi_left < 0 and psi_D > 0) or (psi_left > 0 and psi_D < 0):
        right = D
    else:
        left = D
    
    err = psi_D
    step = step + 1


for i in range(0,step):
    plt.plot(x[i],res[i],alpha=i/step,color='b')

x_ = np.array(x[step-1])
plt.plot(x_,x_**3+2,'r')

plt.xlim(left=a,right=b)
#plt.show()

occuracy = 0
for i in range(0,len(x[step-1])):
    if abs(res[step-1][i] - ((x[step-1][i])**3+2)) > occuracy:
        occuracy = abs(res[step-1][i] - ((x[step-1][i])**3+2))

with open("data.txt", "a") as myfile:
    myfile.write(str(h) + "\t" + str(occuracy)+'\n')