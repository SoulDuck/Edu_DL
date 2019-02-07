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


def var_summary(x):
    ret_dict = {}
    if isinstance(x, np.ndarray):
        ret_dict['mean'] = x.mean()
        ret_dict['std'] = x.std()
        ret_dict['min'] = x.min()
        ret_dict['max'] = x.max()
    else:
        raise ValueError
    return ret_dict

# Train Summaries
train_xs_0_summaries = var_summary(train_xs[:,0])
train_xs_1_summaries = var_summary(train_xs[:,1])

# Test Summaries
test_xs_0_summaries = var_summary(test_xs[:,0])
test_xs_1_summaries = var_summary(test_xs[:,1])

# Validation Summaries
val_xs_0_summaries = var_summary(val_xs[:,0])
val_xs_1_summaries = var_summary(val_xs[:,1])

#
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(331)

train_colors = ['b' if label == 0 else 'r' for label in train_ys]
ax.scatter(train_xs[:,0], train_xs[:,1], label='Train data', alpha=0.5, marker='o', c=train_colors)
ax.set_xlabel('Tumor size')
ax.set_ylabel('age')
ax.set_xlim(0,10)
ax.set_ylim(0,10)
ax.set_title('train')
ax.legend()

ax = fig.add_subplot(332)
ax.bar(list(train_xs_0_summaries.keys()), list(train_xs_0_summaries.values()), align='center', color='b')
ax.set_title('age')

ax = fig.add_subplot(333)
ax.bar(list(train_xs_1_summaries.keys()), list(train_xs_1_summaries.values()), align='center', color='r')
ax.set_title('tumor')


ax = fig.add_subplot(334)
test_colors = ['b' if label == 0 else 'r' for label in test_ys]
ax.scatter(test_xs[:,0], test_xs[:,1], label='Test data', alpha=0.5, marker='+', c=test_colors)
ax.set_xlabel('Tumor size')
ax.set_ylabel('age')
ax.set_xlim(0,10)
ax.set_ylim(0,10)
ax.set_title('test')
ax.legend()

ax = fig.add_subplot(335)
ax.bar(list(test_xs_0_summaries.keys()), list(test_xs_0_summaries.values()), align='center', color='b')

ax = fig.add_subplot(336)
ax.bar(list(test_xs_1_summaries.keys()), list(test_xs_1_summaries.values()), align='center', color='r')




ax = fig.add_subplot(337)
val_colors = ['b' if label == 0 else 'r' for label in val_ys]
ax.scatter(val_xs[:,0], val_xs[:,1], label='Validation data', alpha=0.5, marker='*', c=val_colors)
ax.set_xlabel('Tumor size')
ax.set_ylabel('age')
ax.set_xlim(0,10)
ax.set_ylim(0,10)
ax.legend()
ax.set_title('Validation')

ax = fig.add_subplot(338)
ax.bar(list(val_xs_0_summaries.keys()), list(val_xs_0_summaries.values()), align='center', color='b')

ax = fig.add_subplot(339)
ax.bar(list(val_xs_1_summaries.keys()), list(val_xs_1_summaries.values()), align='center', color='r')

plt.tight_layout()
plt.show()

