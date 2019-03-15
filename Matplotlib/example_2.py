# scatter

import matplotlib.pyplot as plt

plt.scatter(x=[1,2,3,4,5],y=[2,4,6,8,10],s=30,c='b',alpha=0.5,marker='.',label='example')
plt.title('Scatter Example')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0,5)
plt.ylim(0,12)
plt.legend()
plt.show()


plt.scatter(x=[1,2,3,4,5],y=[2,4,6,8,10],s=30,c='b',alpha=0.5,marker='.',label='example')
plt.scatter(x=[1,2,3,4,5],y=[8,9,10,11,12],s=30,c='r',alpha=0.5,marker='.',label='example_1')
plt.title('Scatter Example')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0,5)
plt.ylim(0,12)
plt.legend()
plt.show()