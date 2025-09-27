import matplotlib.pyplot as plt
import numpy as np
import inspect
import sys
# sys.path.append('/home/pavel/.local/lib/python3.10/site-packages/vm-2022.11.1.dist-info')
plt.style.use('default')

# import vm
# import vm.difs as difs


##############
def left_difference(n: int, h: float, derp: list[float], funp: list[float]) -> None:
    res = [None] * n
    for q in range(n-1):
        res[q] = (funp[q+1] - funp[q])/h

    res[n-1] = (funp[n-1] - funp[n-2])/h
    return res


def right_difference(n: int, h: float, derp: list[float], funp: list[float]) -> None:
    res = [None] * n
    res[0] = (funp[1] - funp[0])/h

    for q in range(1, n):
        res[q] = (funp[q] - funp[q-1])/h

    return res


def symmetric_difference(n: int, h: float, derp: list[float], funp: list[float]) -> None:
    res = [None] * n
    res[0] =   (4*funp[1] - 3*funp[0] - funp[2])/(2*h)
    res[n-1] = (3*funp[n-1] - 4*funp[n-2] + funp[n-3])/(2*h)

    for q in range(1, n-1):
        res[q] = (funp[q+1] - funp[q-1])/(2*h)

    return res


def second_difference(n: int, h: float, derp: list[float], funp: list[float]) -> None:
    res = [None] * n
    res[0] =   (2*funp[0] - 5*funp[1] + 4*funp[2] -funp[3])/(h*h)
    res[n-1] = (-funp[n-4] + 4*funp[n-3] - 5*funp[n-2] + 2*funp[n-1])/(h*h)

    for q in range(1, n-1):
        res[q] = (funp[q+1] - 2*funp[q] + funp[q-1])/(h*h)

    return res

##############

fl = [
    lambda x: x**2,
    lambda x: -np.sin(1/5*x),
    lambda x: 0.1*x*np.exp(np.sin(x)),
    lambda x: 0.007*x**2*np.arctan(x)+2,
    lambda x: np.sin(x)/x
]
dl = [
    lambda x: 2*x,
    lambda x: -np.cos(1/5*x)/5,
    lambda x: 0.1*np.exp(np.sin(x)) + 0.1*x*np.exp(np.sin(x))*np.cos(x),
    lambda x: 0.014*x*np.arctan(x) + 0.007*x**2/(1+x**2),
    lambda x: np.cos(x)/x - np.sin(x)/x**2
]



#############FIRST DERIVATIVE
a, b = -3, 5
count = 20
# x = np.linspace(a, b, count)
# y = np.apply_along_axis(f, 0, x)

# d = left_difference(count, (b-a)/(count-1), list(x), list(y))
# d = right_difference(count, (b-a)/(count-1), list(x), list(y))
# d = symmetric_difference(count, (b-a)/(count-1), list(x), list(y))

for f, d in zip(fl, dl):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    x = np.linspace(a, b, count)
    fy = np.apply_along_axis(f, 0, x)
    ax[0].set_title(inspect.getsource(f)[14:-1])
    ax[0].plot(x, fy)
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].grid()

    x = np.linspace(a, b, count)
    dy = np.apply_along_axis(d, 0, x)
    d1 = left_difference(count, (b-a)/(count-1), list(x), list(fy))
    d2 = right_difference(count, (b-a)/(count-1), list(x), list(fy))
    d3 = symmetric_difference(count, (b-a)/(count-1), list(x), list(fy))
    # d3 = second_difference(count, (b-a)/(count-1), list(x), list(fy))
    ax[1].set_title(inspect.getsource(d)[14:-1])
    ax[1].plot(x, dy)
    ax[1].scatter(x, dy)
    ax[1].plot(x, d1, label='left')
    ax[1].plot(x, d2, label='rigth')
    ax[1].plot(x, d3, label='symmetric')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y\'')
    ax[1].legend()
    ax[1].grid()

    d1_ = []
    d2_ = []
    d3_ = []
    points = np.arange(5, 400)
    for q in points:
        x = np.linspace(a, b, q)
        fy = np.apply_along_axis(f, 0, x)
        dy = np.apply_along_axis(d, 0, x)
        d1 = left_difference(     q, (b-a)/(q-1), list(x), list(fy))
        d2 = right_difference(    q, (b-a)/(q-1), list(x), list(fy))
        d3 = symmetric_difference(q, (b-a)/(q-1), list(x), list(fy))
        d1_.append(np.max(np.abs(d1-dy)))
        d2_.append(np.max(np.abs(d2-dy)))
        d3_.append(np.max(np.abs(d3-dy)))
    ax[2].set_title('$log(\epsilon)(log(h))$')
    ax[2].plot(np.log((b-a)/(points-1)), np.log(d1_), label='left')
    ax[2].plot(np.log((b-a)/(points-1)), np.log(d2_), label='rigth')
    ax[2].plot(np.log((b-a)/(points-1)), np.log(d3_), label='symmetric')
    ax[2].set_xlabel('$log(h)$')
    ax[2].set_ylabel('$log(\epsilon)$')
    ax[2].legend()
    ax[2].grid()

    plt.savefig("first-der.png")
    # break
    # ax[0].set_ylabel('y')



############MACHINE PRECISION
count = 15
points = 2**(np.arange(2, count))
a, b = -2.002, -2.001
fn = 2

d2_ = []
d3_ = []
for q in points:
    x = np.linspace(a, b, q)
    fy = np.apply_along_axis(fl[fn], 0, x)
    dy = np.apply_along_axis(dl[fn], 0, x)

    # d2 = right_difference(    q, (b-a)/(q-1), list(x), list(fy))
    d3 = symmetric_difference(q, (b-a)/(q-1), list(x), list(fy))
    # d2_.append(np.max(np.abs(d2-dy)))
    d3_.append(np.max(np.abs(d3-dy)))
    
# plt.plot((b-a)/(points-1), d2_)
# plt.plot((b-a)/(points-1), d3_)
# plt.plot(points, d3_)
plt.figure(figsize=(10, 7))
plt.title(inspect.getsource(fl[fn])[14:-1])
plt.plot(np.arange(2, count), d3_)
plt.xlabel('power of 2')
plt.ylabel('$\epsilon$')
plt.grid()

plt.savefig("machine-prec.png")








###########SECOND DERIVATIVE

a, b = -1, 1

func0d = lambda x: x**6 + np.sin(x*10)
func1d = lambda x: 6*x**5 + 10*np.cos(x*10)
func2d = lambda x: 30*x**4 - 100*np.sin(x*10)
# func0d = lambda x: x**6
# func1d = lambda x: 6*x**5
# func2d = lambda x: 30*x**4

x = np.linspace(a, b, 100)
f = func0d(x)
d = func2d(x)
d4 = second_difference(len(x), (b-a)/(len(x)-1), list(x), list(f))

points = np.arange(5, 400)
err = []
for q in points:
    xl = np.linspace(a, b, q)
    fl = func0d(np.linspace(a, b, q))
    dl = func2d(np.linspace(a, b, q))
    er = np.max(np.abs(
        second_difference(q, (b-a)/(q-1), list(xl), list(fl)) - dl
    ))
    err.append(er)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].plot(x, func0d(x))
ax[0].set_title('$x^6 + sin(10x)$')

ax[1].scatter(x, func2d(x))
ax[1].set_title('$30*x^4 - 100cos(10x)$')
ax[1].plot(x, d4)

ax[2].plot(np.log((b-a)/(points-1)), np.log(err))
ax[2].set_title('$log\epsilon(logh)$')
ax[2].set_xlabel('$log(h)$')
ax[2].set_ylabel('$log(\epsilon)$')

plt.savefig("second-der.png")
