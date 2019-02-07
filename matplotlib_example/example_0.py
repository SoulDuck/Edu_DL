import matplotlib.pyplot as plt
import numpy as np

xs = [1,2,3,4]
ys = [3,4,5,6]
#case 0
plt.plot(xs, ys, c='b', marker='.', linestyle='--')
plt.xlabel('case1 x axis')
plt.ylabel('case1 y aixs')
plt.show()


#case 1
plt.plot(xs, ys, 'r,-.')
plt.xlabel('case2 x axis')
plt.ylabel('case2 y aixs')
plt.show()


#case 2
plt.plot(xs, ys, 'r,-.')
plt.xlabel('case2 x axis')
plt.ylabel('case2 y aixs')
plt.show()

#case 3
plt.plot(xs, ys, 'yv-.', label='main')
plt.title('CASE 2')
plt.xlabel('case2 x axis')
plt.ylabel('case2 y aixs')
plt.legend()
plt.show()

#case 4
plt.plot(xs, ys, 'yv-.', label='main')
plt.title('CASE 2')
plt.xlabel('case2 x axis')
plt.ylabel('case2 y aixs')
plt.xlim(0,10)
plt.ylim(0,10)
plt.legend()
plt.show()










