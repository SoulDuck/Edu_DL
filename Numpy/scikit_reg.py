
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

datum = np.genfromtxt('cancer_data.csv',delimiter=',',dtype=np.float32, skip_header=True)
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
test_y = Y[60:80]
val_y = Y[80:]

regr = linear_model.LinearRegression()
regr.fit(train_x, train_y)
test_preds = regr.predict(test_x)

fig = plt.figure()
ax = fig.add_subplot(2,2,1)
ax.set_xlim(0,10)
ax.set_ylim(0,10)
ax.set_title('Pred')
test_preds = [0 if pred <= 0.5 else 1 for pred in test_preds]
print(test_preds )
pred_colors = ['b' if pred ==0 else 'r' for pred in test_preds]
print(pred_colors )
ax.scatter(test_x[:,0], test_x[:,1], color=pred_colors, linewidth=3)


ax = fig.add_subplot(2,2,2)
ax.set_xlim(0,10)
ax.set_ylim(0,10)
ax.set_title('True')
y_colors = ['b' if y==0 else 'r' for y in test_y]
ax.scatter(test_x[:,0], test_x[:,1], color=y_colors, linewidth=3)
plt.show()

acc = np.mean(np.equal(test_preds,test_y))
print(acc)


