import os
import numpy as np
import keras
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,BatchNormalization
from keras.layers import Conv2D,Convolution2D,  MaxPooling2D, ZeroPadding2D
from keras.applications.vgg16 import VGG16
from keras.utils import np_utils
from keras.datasets import mnist



# load data from files
X_train = np.load('xtrain.npy')
X_test = np.load('xtest.npy')
Y_train = np.load('ytrain.npy')
print("Finished loading data")


act = 'sigmoid'
batch_size = 128
num_classes =3
filter_pixel=3
noise = 1
droprate=0.25
l2_lambda = 0.0001
reg = l2(l2_lambda)

# input image dimensions
img_rows, img_cols = 28, 28

input_shape=X_train.shape[1:]
#Start Neural Network
model = Sequential()
#convolution 1st layer
model.add(Conv2D(64, kernel_size=(filter_pixel, filter_pixel),
                 activation=act,
		 kernel_regularizer=reg,
                 input_shape=input_shape)) #0
model.add(BatchNormalization())
model.add(Dropout(droprate))#3
model.add(MaxPooling2D())

#convolution 2nd layer
model.add(Conv2D(64, kernel_size=(filter_pixel, filter_pixel), kernel_regularizer=reg,activation=act))#1
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(droprate))#3

#convolution 3rd layer
model.add(Conv2D(64, kernel_size=(filter_pixel, filter_pixel),kernel_regularizer=reg, activation=act))#1
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(droprate))#3

#Fully connected 1st layer
model.add(Flatten()) #7
model.add(Dense(218,kernel_regularizer=reg)) #13
model.add(BatchNormalization())
model.add(Activation(act)) #14
model.add(Dropout(droprate))      #15

#Fully connected final layer
model.add(Dense(num_classes)) #8
model.add(Activation('softmax')) #9


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

print(model.summary())

callbacks = [keras.callbacks.EarlyStopping(patience=20,verbose=1)]
model.fit(X_train, Y_train,validation_split = 0.2,batch_size=75,epochs=100,verbose=1)
print("Finished Fitting Model")
predict = model.predict(X_test)
np.save('predict.npy',predict)


#os.remove("sub.csv")
#print('Prev file deleted')
file = open("sub.csv ","w")
file.write( "Id,Category\n")
for i in range(len(predict)):
   # print(predict[i])
    file.write( "%i,%i\n" % (i, predict[i].argmax()))
file.close()

