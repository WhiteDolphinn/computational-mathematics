import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

def create_method_by_table(table, ts):
    def table_method(fn, table, ts, h, n, x0, y0):
        arr = np.arange(n, dtype=np.float64)
        arr[0] = y0
        
        for e in range(n-1):
            k = np.zeros(ts-1)
            for q in range(ts-1):
                x_ = x0 + e*h + h*table[q][0]
                y_ = 0
                for w in range(q):
                    y_ += table[q][w]*k[w]
                y_ = arr[e] + h*y_

                k[q] = fn(x_, y_)

            arr[e+1] = arr[e] + h*np.sum(table[-1][1:]@k.T)

        return arr
    
    def table_method_sys(fn, table, ts, h, n, m, x0, y0):
        arr = np.zeros((n, m), dtype=np.float64)
        arr[0] = y0
        
        for e in range(n-1):
            k = np.zeros((ts-1, m), dtype=np.float64)
            for q in range(ts-1):
                x_ = x0 + e*h + h*table[q][0]
                y_ = np.zeros(m)
                for w in range(q):
                    y_ += table[q][w]*k[w]
                y_ = arr[e] + h*y_
                
                for r in range(m):
                    k[q][r] = (fn[r])(x_, y_)
                    
            arr[e+1] = arr[e] + h*np.sum(
                        [t*k_ for t, k_ in zip(table[-1][1:], k)], axis=0)

        return arr
    
    # return lambda fn, h, n, x0, y0: table_method(fn, table, ts, h, n, x0, y0)
    return lambda fn, h, n, m, x0, y0: table_method_sys(fn, table, ts, h, n, m, x0, y0)

def fn_populationGrow(x, y):
    return y-y**2

def fn(x, y):
    return -x*(y**4+1)/2/(x*y+y)

# def Eulerian_method(x, y):
#     return table_method(
#         fn_populationGrow,
#         [[0,0], [0, 1]],
#         2, h, n, x0, y0
#     )

# def Eulerian_method_with_recalculation(x, y):
#     return table_method(
#         fn_populationGrow,
#         [[0,0,0], [0.5,0.5, 1], [0,0,1]],
#         3, h, n, x0, y0
#     )

# def Hoyne_method(x, y):
#     return table_method(
#         fn_populationGrow,
#         [[0,0,0,0], [1/3,1/3,0,0], [2/3,0,2/3,0], [0,1/4,0,3/4]],
#         4, h, n, x0, y0
#     )

# def third_degree_Runge_Kutta_method(x, y):
#     return table_method(
#         fn_populationGrow,
#         [[0,0,0,0], [1/2,1/2,0,0], [1,0,1,0], [0,1/6,2/3,1/6]],
#         4, h, n, x0, y0
#     )

# def Runge_Kutta_method(x, y):
#     return table_method(
#         fn_populationGrow,
#         [[0,0,0,0,0], [1/2,1/2,0,0,0], [1/2,0,1/2,0,0], [1,0,0,1,0],
#                                                         [0,1/6,2/6,2/6,1/6]],
#         5, h, n, x0, y0
#     )

# def three_eighths_rule(x, y):
#     return table_method(
#         fn_populationGrow,
#         [[0,0,0,0,0], [1/3,1/3,0,0,0], [2/3,-1/3,1,0,0], [1,1,-1,1,0],
#                                                         [0,1/8,3/8,3/8,1/8]],
#         5, h, n, x0, y0
#     )

# def Butcher_method(x, y):
#     return table_method(
#         fn_populationGrow,
#         [[0,0,0,0,0], [1/3,1/3,0,0,0], [2/3,-1/3,1,0,0], [1,1,-1,1,0],
#                                                         [0,1/8,3/8,3/8,1/8]],
#         5, h, n, x0, y0
#     )


Eulerian_method = create_method_by_table(
    [
        [0,0],
        [0, 1]
    ], 2
)

Eulerian_method_with_recalculation = create_method_by_table(
    [
        [0,0,0],
        [0.5,0.5, 1],
        [0,0,1]
    ], 3
)

Hoyne_method = create_method_by_table(
    [
        [0,0,0,0],
        [1/3,1/3,0,0],
        [2/3,0,2/3,0],
        [0,1/4,0,3/4]
    ], 4
)

third_degree_Runge_Kutta_method = create_method_by_table(
    [
        [0,0,0,0],
        [1/2,1/2,0,0],
        [1,0,1,0],
        [0,1/6,2/3,1/6]
    ], 4
)

Runge_Kutta_method = create_method_by_table(
    [
        [0,0,0,0,0],
        [1/2,1/2,0,0,0],
        [1/2,0,1/2,0,0],
        [1,0,0,1,0],
        [0,1/6,2/6,2/6,1/6]
    ], 5
)

three_eighths_rule = create_method_by_table(
    [
        [0,0,0,0,0],
        [1/3,1/3,0,0,0],
        [2/3,-1/3,1,0,0],
        [1,1,-1,1,0],
        [0,1/8,3/8,3/8,1/8]
    ], 5
)

Butcher_method = create_method_by_table(
    [
        [0,0,0,0,0,0,0,0],
        [1/2,1/2,0,0,0,0,0,0],
        [2/3,2/9,4/9,0,0,0,0,0],
        [1/3,7/36,2/9,-1/12,0,0,0,0],
        [5/6,-35/144,-55/36,35/48,15/8,0,0,0],
        [1/6,-1/360,-11/36,-1/8,1/2,1/10,0,0],
        [1,-41/260,22/13,43/156,-118/39,32/195,80/39,0],
        [0,13/200,0,11/40,11/40,4/25,4/25,13/200]
    ], 8
)


def fn(x, y):
    return y-y**2

n = 8000
h = 0.001
x0, y0 = 0, 0.2
arr = Eulerian_method([fn], h, n, 1, x0, y0)

plt.plot(x0+np.arange(n)*h, arr)
plt.savefig("Eulerian.png")

def fn(x, y):
    return -x*(y**2+1)/2/(x+y)

n = 300
h = 0.01
x0, y0 = -0.8, np.sqrt(np.tan(1.5+np.log(np.abs(x0+1))-x0))
arr = Eulerian_method([fn], h, n, 1, x0, y0)
arr2 = Hoyne_method([fn], h, n, 1, x0, y0)
arr3 = Butcher_method([fn], h, n, 1, x0, y0)


plt.figure(figsize=(5, 5))
plt.plot(x0+np.arange(n)*h, arr, color='r')
plt.plot(x0+np.arange(n)*h, arr2, color='g')
plt.plot(x0+np.arange(n)*h, arr3, color='b')
plt.savefig("Eulerian2.png")

# def calc_speed(meth):
#     def fn(x, y):
#         return y-y**2

#     n = 6000
#     h = 0.001
#     b = np.arange(1,11)
#     nl = 300*b
#     hl = 0.01/b
#     x0, y0 = 0, np.array([1/2])
#     err_arr = []
#     for n, h in zip(nl, hl): 
#         y = meth([fn], h, n, 1, x0, y0).T[:2]
#         x = x0 + np.arange(n)*h
#         err = np.max((1/(1+np.exp(-x)) - y)**2)
#         err_arr.append(err)

#     plt.plot(np.log(hl), np.log(err_arr))
#     k = linregress(np.log(hl), np.log(err_arr))
#     print(k[0])

# calc_speed(Eulerian_method)
# calc_speed(Hoyne_method)
# calc_speed(Butcher_method)


# plt.scatter(*y0, color='black')
plt.plot(*Eulerian_method([fn], h, n, 1, x0, y0).T)
plt.plot(*Hoyne_method([fn], h, n, 1, x0, y0).T)
plt.plot(*Butcher_method([fn], h, n, 1, x0, y0).T)
plt.legend([
    'Eulerian_method',
    'Hoyne_method',
    'Butcher_method'
])
plt.savefig("all_equation.png")






fn = [
    lambda x, r: 1,
    lambda x, r: x,
]

n = 6000
h = 0.001
x0, y0 = 0, np.array([1,0])
arr = Eulerian_method(
    fn,
    h, n, 2, x0, y0
)

plt.figure(figsize=(5, 5))
plt.plot(*arr[:, :2].T[:2])
plt.savefig("Eulerian3.png")



fn = [
    lambda x, r: r[2],
    lambda x, r: r[3],
    lambda x, r: -r[0],
    lambda x, r: -r[1],
]

n = 6000
h = 0.001
x0, y0 = 0, np.array([1,0,0,1])
arr = Hoyne_method(
    fn,
    h, n, 4, x0, y0
)

plt.figure(figsize=(5, 5))
plt.plot(*arr[:, :2].T)
plt.savefig("Hoyne.png")


a = 3
b = 4
c = 5

fn = [
    lambda t, x: x[1] + x[2] - a*x[0],
    lambda t, x: x[2] + x[0] - b*x[1],
    lambda t, x: x[0] + x[1] - c*x[2],
    ]


n = 6000
h = 0.01
x0, y0 = 0, np.array([1,1,0])
arr = Butcher_method(
    fn,
    h, n, 3, x0, y0
)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
plt.plot(*arr.T)
plt.savefig("Butcher.png")


a = 0.1
b = 0.1
c = 14

fn = [
    lambda t, x: -x[1] - x[2],
    lambda t, x: x[0] + a*x[1],
    lambda t, x: b + x[2]*(x[0] - c),
    ]
# fn = [
#     lambda t, x: x[1] + x[2] - a*x[0],
#     lambda t, x: x[2] + x[0] - b*x[1],
#     lambda t, x: x[0] + x[1] - c*x[2],
#     ]

n = 6000
h = 0.01
x0, y0 = 0, np.array([1,1,0])
arr = Butcher_method(
    fn,
    h, n, 3, x0, y0
)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
plt.plot(*arr.T)
plt.savefig("Butcher2.png")


a = 1.4
fn = [
    lambda t, x: -a*x[0] - 4*x[1] - 4*x[2] - x[1]**2,
    lambda t, x: -a*x[1] - 4*x[2] - 4*x[0] - x[2]**2,
    lambda t, x: -a*x[2] - 4*x[0] - 4*x[1] - x[0]**2,
]

n = 10000
h = 0.01
x0, y0 = 0, np.array([0,0,1])
# arr = Butcher_method(
arr = Eulerian_method_with_recalculation(
    fn,
    h, n, 3, x0, y0
)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
plt.plot(*arr.T)


# Расходимость
a = 1.4
fn = [
    lambda t, x: -a*x[0] - 4*x[1] - 4*x[2] - x[1]**2,
    lambda t, x: -a*x[1] - 4*x[2] - 4*x[0] - x[2]**2,
    lambda t, x: -a*x[2] - 4*x[0] - 4*x[1] - x[0]**2,
]

n = 200
h = 0.01
x0, y0 = 0, np.array([0,0,1])
# x0, y0 = 0, np.array([0,0])
arr = Butcher_method(fn, h, n, 3, x0, y0)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
plt.scatter(*y0, color='black')
plt.plot(*Eulerian_method(fn, h, n, 3, x0, y0).T)
plt.plot(*Eulerian_method_with_recalculation(fn, h, n, 3, x0, y0).T)
plt.plot(*Hoyne_method(fn, h, n, 3, x0, y0).T)
plt.plot(*third_degree_Runge_Kutta_method(fn, h, n, 3, x0, y0).T)
plt.plot(*Runge_Kutta_method(fn, h, n, 3, x0, y0).T)
plt.plot(*three_eighths_rule(fn, h, n, 3, x0, y0).T)
plt.plot(*Butcher_method(fn, h, n, 3, x0, y0).T)
plt.legend([
    'Eulerian_method',
    'Eulerian_method_with_recalculation',
    'Hoyne_method',
    'third_degree_Runge_Kutta_method',
    'Runge_Kutta_method',
    'three_eighths_rule',
    'Butcher_method'
])
plt.savefig("all.png")

# Скорость сходимости
plt.figure(figsize=(5, 5))
plt.legend([
    'Eulerian_method',
    'Eulerian_method_with_recalculation',
    'Hoyne_method',
    'third_degree_Runge_Kutta_method',
    'Runge_Kutta_method',
    'three_eighths_rule',
    'Butcher_method'
])
def calc_speed(meth):
    def fn(x, y):
        return y-y**2

    n = 6000
    h = 0.001
    b = np.arange(1,11)
    nl = 300*b
    hl = 0.01/b
    x0, y0 = 0, np.array([1/2])
    err_arr = []
    for n, h in zip(nl, hl): 
        y = meth([fn], h, n, 1, x0, y0).T[:2]
        x = x0 + np.arange(n)*h
        err = np.max((1/(1+np.exp(-x)) - y)**2)
        err_arr.append(err)

    plt.plot(np.log(hl), np.log(err_arr))
    k = linregress(np.log(hl), np.log(err_arr))
    print(k[0])

calc_speed(Eulerian_method)
calc_speed(Eulerian_method_with_recalculation)
calc_speed(Hoyne_method)
calc_speed(third_degree_Runge_Kutta_method)
calc_speed(Runge_Kutta_method)
calc_speed(three_eighths_rule)
calc_speed(Butcher_method)
plt.savefig("speed.png")