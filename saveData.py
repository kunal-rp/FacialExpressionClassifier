import numpy as np
import pandas as pd
import keras


X_test = pd.read_csv('test_data.csv',header=None).as_matrix()
X_train = pd.read_csv('train_data.csv',header=None).as_matrix()
Y_train = pd.read_csv('train_target.csv',header=None)

X_test = np.array([n.reshape(48,48) for n in X_test])
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)


X_train = np.array([n.reshape(48,48) for n in X_train])
X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)


Y_train = keras.utils.np_utils.to_categorical(Y_train,num_classes=3)
print(X_test.shape)
print(X_train.shape)
print(Y_train.shape)
np.save('xtrain.npy',X_train)
np.save('xtest.npy',X_test)
np.save('ytrain.npy',Y_train)
