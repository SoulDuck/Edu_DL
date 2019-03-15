import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt


datum = np.genfromtxt('cancer_data.csv',dtype=np.float32, delimiter=',')
datum = datum[1:,:]
npr.shuffle(datum)
xs = datum[:,:2]
ys = datum[:,-1]

# train
train_xs = xs[:60]
train_ys = ys[:60]

# test
test_xs = xs[60:80]
test_ys = ys[60:80]

# train
val_xs = xs[80:]
val_ys = ys[80:]

print('Train shape : {} \t Test Shape : {} \t Validation shape : {}'.format(
    train_xs.shape, test_xs.shape, test_xs.shape)
)


train_colors = ['b' if label == 0 else 'r' for label in train_ys]
plt.scatter(train_xs[:,0], train_xs[:,1], label='Train data', alpha=0.5, marker='o', c=train_colors)

test_colors = ['b' if label == 0 else 'r' for label in test_ys]
plt.scatter(test_xs[:,0], test_xs[:,1], label='Test data', alpha=0.5, marker='+', c=test_colors)

val_colors = ['b' if label == 0 else 'r' for label in val_ys]
plt.scatter(val_xs[:,0], val_xs[:,1], label='Validation data', alpha=0.5, marker='*', c=val_colors)

plt.xlabel('Tumor size')
plt.ylabel('Age')
plt.legend()
plt.show()
