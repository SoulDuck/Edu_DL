from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open('cifar_horse.png').resize([32,32]).convert('RGB')
np_img = np.asarray(img)
red_img = np_img[:,:,0]
blue_img = np_img[:,:,1]
green_img = np_img[:,:,2]

# Extract Red Color
red_img = np_img.copy()
red_img[:,:,1]=0
red_img[:,:,2]=0

green_img = np_img.copy()
green_img[:,:,0]=0
green_img[:,:,2]=0

blue_img = np_img.copy()
blue_img[:,:,0]=0
blue_img[:,:,1]=0
msg=''

for row in blue_img[:,:,2]:
    for e in row:
        msg += str(e)+'\t'
    msg += '\n'
print(msg)

plt.imshow('horse_r.png')
plt.show()
plt.close()

plt.imshow('horse_g.png')
plt.show()
plt.close()

plt.imshow('horse_b.png')
plt.show()
plt.close()









