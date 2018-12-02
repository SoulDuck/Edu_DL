import sys
sys.path.append('../')
sys.path.append('../../')


def plot_images(imgs, names=None, random_order=False, savepath=None, no_axis=True):
    import math
    import matplotlib.pyplot as plt
    import random
    h = math.ceil(math.sqrt(len(imgs)))
    fig = plt.figure()
    for i in range(len(imgs)):
        ax = fig.add_subplot(h, h, i + 1)
        if random_order:
            ind = random.randint(0, len(imgs) - 1)
        else:
            ind = i
        img = imgs[ind]
        plt.axis('off')
        plt.imshow(img)
        if not names == None:
            ax.set_xlabel(names[ind])
    if not savepath is None:
        plt.savefig(savepath)
    plt.show()