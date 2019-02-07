import matplotlib.pyplot as plt
import autograd.numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from matplotlib import animation
from IPython.display import HTML

from autograd import elementwise_grad, value_and_grad
from scipy.optimize import minimize
from collections import defaultdict
from itertools import zip_longest
from functools import partial

f  = lambda x, y: (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

xmin, xmax, xstep = -4.5, 4.5, .2
ymin, ymax, ystep = -4.5, 4.5, .2

x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
z = f(x, y)

minima = np.array([3., .5]) # minima 인 곳을 별로 표시함
f(*minima) # minima value : 0

minima_ = minima.reshape(-1, 1)


fig = plt.figure(figsize=(8, 5))
ax = plt.axes(projection='3d', elev=50, azim=-50)

ax.plot(*minima_, f(*minima_), 'r*', markersize=10)
ax.plot_surface(x, y, z, norm=LogNorm(), rstride=1, cstride=1,
                edgecolor='c', alpha=0.5, cmap=plt.cm.jet, linewidth=0.5)

ax.view_init(30, 10)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')

ax.set_xlim((xmin, xmax))
ax.set_ylim((ymin, ymax))

plt.show()
plt.close()

# Optimizer
x0 = np.array([3., 4.])
func = value_and_grad(lambda args: f(*args))


def make_minimize_cb(path=[]):
    def minimize_cb(xk):
        # note that we make a deep copy of xk
        path.append(np.copy(xk))

    return minimize_cb

path_ = [x0]
res = minimize(func, x0=x0, method='Newton-CG',
               jac=True, tol=1e-20, callback=make_minimize_cb(path_))

print(path_)
path_ = np.asarray(path_)
path = path_.T

fig = plt.figure(figsize=(8, 5))
ax = plt.axes(projection='3d', elev=50, azim=-50)

# logNorm -> Normalize a given value to the 0-1 range on a log scale
ax.plot_surface(x, y, z, norm=LogNorm(), rstride=2, cstride=2, edgecolor='none', alpha=0.5, cmap=plt.cm.jet)
ax.quiver(path[0,:-1], path[1,:-1], f(*path[::,:-1]),
          path[0,1:] - path[0,:-1], path[1,1:] - path[1,:-1], f(*(path[::,1:] - path[::,:-1])),
          color='k')

ax.plot(*minima_, f(*minima_), 'r*', markersize=10)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')

ax.set_xlim((xmin, xmax))
ax.set_ylim((ymin, ymax))

plt.show()
plt.close()


