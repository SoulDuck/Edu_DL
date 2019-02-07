import numpy as np
import numpy.random as npr
data = np.arange(0,9,1)
print(data)
#s = slice(2,7,2)
print(data[2:5])
print(data[2:])
print(data[2:5:2])


datum = np.genfromtxt('cancer_data.csv', delimiter=',', dtype=np.string_)
# except header
datum = datum[1:,:]

# Shuffle
npr.shuffle(datum)

X = datum[:, :2]
Y = datum[:, 2]



train_x = X[:60]
test_x = X[60:80]
val_x = X[80:]

train_y= Y[:60]
test_Y = Y[60:80]
val_Y = Y[80:]


print(train_x[3])
print(train_x[5])
print(train_x[7])
print(train_x[11])
print(train_x[[3,5,7]])
print(train_x.dtype)
print(test_x.dtype)

print(np.asarray(train_x, np.float32))
train_x = train_x.astype(np.float32)
print(train_x.mean())
print(train_x.min())
print(train_x.max())
print(train_x.std())

test_x = test_x.astype(np.float32)
print(test_x.mean())
print(test_x.min())
print(test_x.max())
print(test_x.std())


val_x = val_x.astype(np.float32)
print(val_x.mean())
print(val_x.min())
print(val_x.max())
print(val_x.std())

print('\n')
print(train_x[:,0].mean())
print(train_x[:,0].min())
print(train_x[:,0].max())
print(train_x[:,0].std())

print('\n')
print(train_x[:,1].mean())
print(train_x[:,1].min())
print(train_x[:,1].max())
print(train_x[:,1].std())


print('\n')
print(val_x[:,0].mean())
print(val_x[:,0].min())
print(val_x[:,0].max())
print(val_x[:,0].std())

print('\n')
print(val_x[:,1].mean())
print(val_x[:,1].min())
print(val_x[:,1].max())
print(val_x[:,1].std())

print('\n')
print(test_x[:,0].mean())
print(test_x[:,0].min())
print(test_x[:,0].max())
print(test_x[:,0].std())

print('\n')
print(test_x[:,1].mean())
print(test_x[:,1].min())
print(test_x[:,1].max())
print(test_x[:,1].std())


"""

from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X, Y)
print(reg)
"""


indices = np.where(train_x[:,0]>train_x.mean())[0]
print(np.where(train_x[:,0]>train_x.mean()))
print(train_x[indices])
print(train_x.mean())