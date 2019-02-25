import numpy as np
import matplotlib.pyplot as plt

def time_series(t):
    return t * np.sin(t) / 3 + 2 * np.sin(t * 5)


a = np.arange(0, 30, 0.1)
b = time_series(a)

plt.plot(a, b, label='t * sin(t) / 3 + 2 * sin(t * 5)')
plt.legend()
plt.show()


plt.scatter(a[:5], b[:5] , c='r')
plt.scatter(a[5], b[5] , c='y')
plt.show()



