# dataset 을 설명하기 위해 임시로 만든 python file
import numpy as np
import matplotlib.pyplot as plt


# Data
def time_series(t):
    return t * np.sin(t) / 3 + 2 * np.sin(t * 5)

xs = np.arange(1, 30, 0.1)
ys = time_series(xs)

xs[#12.2 +  + 1)]
plt.scatter(xs, ys)
plt.show()

