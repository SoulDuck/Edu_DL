# case 1
import numpy as np

a = np.array([[0, 1, 2],[0, 1, 2]],dtype= np.int32)
print(a)
print(a.shape)
print('ndim', a.ndim)
print(a.itemsize) # byte 크기을 알려줍니다
print(a.flags)



a = np.empty(shape=[1, 2], dtype=np.float32)
print(a)

b = np.zeros(shape=[1, 2], dtype=np.int32)
print(b)

b_1 = np.zeros(shape= [2, 2], dtype=np.int32)
print(b_1)


c = np.ones(shape=[1, 2], dtype=np.float64)
print(c)

d = np.full(shape=[2,2] ,dtype=np.float32, fill_value=3)
print(d)

d = np.arange(0,1,0.1)
print(d)

e = np.linspace(0,1,6)
print(e)

f = np.array([1,2],dtype=np.bool_)
print(f)
print(f.itemsize)

g = np.zeros_like(f, dtype=np.float32)
print(g)

h = np.full_like(f, fill_value=4, dtype=np.float32)
print('h',h)


# python list -> ndarray
python_list = [1,2,3]
print(type(python_list))
g = np.asarray([1,2,3])
print(g)
print(type(g))


# image -> ndarray
from PIL import Image
import matplotlib.pyplot as plt
img = Image.open('/Users/seongjungkim/PycharmProjects/edu_dl/sample.png',)
print(img)
# plt.imshow(img)
# plt.show()
np_img = np.asarray(img)
print(np_img)

# csv -> ndarray
f = open('cancer_data.csv','r')
for line in f.readlines():
    print(line)


my_data = np.genfromtxt('cancer_data.csv', delimiter=',', dtype=np.string_)
print(my_data)