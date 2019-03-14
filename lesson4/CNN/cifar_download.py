from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


img = Image.open('cifar_horse.png').convert('HSV')
print(img)
h_img = np.asarray(img).copy()
s_img = np.asarray(img).copy()
v_img = np.asarray(img).copy()

h_img[:, :, 0] = 0
s_img[:, :, 1] = 0
v_img[:, :, 2] = 0

fig = plt.figure()
for i,img in enumerate([h_img, s_img, v_img]):
    ax = fig.add_subplot(1,3,i+1)
    ax.imshow(img)
plt.show()

