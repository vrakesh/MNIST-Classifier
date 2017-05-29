from mnist_classifier import mnist_classifier
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from random import randint
def plot_prediction(network, X_test, y_test, X_test_in):
    fig = plt.figure(figsize=(20,20))
    fig.suptitle(network,fontsize=50)
    for i in range(6):
        j = randint(0,10000)
        ax = fig.add_subplot(1, 6, i+1, xticks=[], yticks=[])
        ax.imshow(X_test_in[j], cmap='gray')
        act = int(y_test[j])
        if(network == 'mlp'):
            y_hat = model.predict(X_test[j].reshape(1,28,28))
        elif(network == 'cnn'):
            y_hat = model.predict(X_test[j].reshape(1,28,28,1))

        pred = int(y_hat[0].tolist().index(1))
        ax.set_title("{} ({})".format(act, pred),
            color=("green" if pred == act else "red"),fontsize=25)
    #plt.show()
    fig.savefig(''.join([network,'.jpg']))

mc = mnist_classifier()
model = mc.get_model('mlp')
model.load_weights('mnist.model.mlp.hdf5')
X_test, y_test = mc.get_test_data()

#
# predict 7 random digits using mlp
plot_prediction('mlp',X_test, y_test, X_test)

X_test_new, _ = mc.reshape(X_test,X_test)
model = mc.get_model('cnn')
model.load_weights('mnist.model.cnn.hdf5')
plot_prediction('cnn', X_test_new, y_test, X_test)
