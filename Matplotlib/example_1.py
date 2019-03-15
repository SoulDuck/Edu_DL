import matplotlib.pyplot as plt
import numpy as np

# evenly sampled time at 200ms intervals
# case 0
t = np.arange(0., 5., 0.2)
plt.plot(t, t, 'r1:', label='linear function')
plt.plot(t, t**2, 'g<--', label='Quadratic function')
plt.plot(t, t**3, 'b>-', label='Cubic function')

plt.xlabel('xs')
plt.ylabel('ys')
plt.xlim(0,5)
plt.ylim(0,150)
plt.title('Graph')
plt.legend()
plt.show()


# case 1
plt.plot(t, t, 'r--',t, t**2, 'bs', t, t**3, 'g^')
plt.xlabel('xs')
plt.ylabel('ys')
plt.xlim(0,5)
plt.ylim(0,150)
plt.title('Graph')
plt.legend()
plt.show()

