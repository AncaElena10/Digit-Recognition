# from keras.models import Sequential
# from keras.layers import Dense, Conv2D, Flatten
# from keras.datasets import mnist
# from keras.utils import to_categorical
# import convert_to_mnist as ctm
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# # from sklearn.model_selection import cross_val_score
# # from sklearn.model_selection import StratifiedKFold
#
# def prepare_data():
#     # download mnist data and split into train and test sets
#     (X_train, y_train), (X_test, y_test) = mnist.load_data()
#
#     #
#     # plt.imshow(X_test[1])
#     # plt.show()
#
#     # reshape data to fit model
#     X_train = X_train.reshape(60000, 28, 28, 1)
#     X_test = X_test.reshape(10000, 28, 28, 1)
#
#     # one-hot encode target column
#     y_train = to_categorical(y_train)
#     y_test = to_categorical(y_test)
#
#     return X_train, X_test, y_train, y_test
#
#
# def prepare_model():
#     # load data
#     X_train, X_test, y_train, y_test = prepare_data()
#
#     # create model
#     # model = None
#     model = Sequential()
#
#     # add model layers
#     # 1
#     model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
#     model.add(Conv2D(32, kernel_size=3, activation='relu'))
#
#     # 2
#     # model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
#     # model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
#
#     # 3
#     # model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
#     # model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
#
#     # 4
#     # model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
#     # model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
#     # model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
#
#     # 5
#     # model.add(Conv2D(64, kernel_size=(1, 1), activation='relu', input_shape=(28, 28, 1)))
#     # model.add(Conv2D(32, kernel_size=(1, 1), activation='relu'))
#
#     # 6
#     # model.add(Conv2D(128, kernel_size=(1, 1), activation='relu', input_shape=(28, 28, 1)))
#     # model.add(Conv2D(64, kernel_size=(1, 1), activation='relu'))
#
#     # 7
#     # model.add(Conv2D(32, kernel_size=(1, 1), activation='relu', input_shape=(28, 28, 1)))
#     # model.add(Conv2D(16, kernel_size=(1, 1), activation='relu'))
#
#     # 8
#     # model.add(Conv2D(64, kernel_size=(1, 1), activation='relu', input_shape=(28, 28, 1)))
#     # model.add(Conv2D(32, kernel_size=(1, 1), activation='relu'))
#     # model.add(Conv2D(16, kernel_size=(1, 1), activation='relu'))
#
#     model.add(Flatten())
#     model.add(Dense(10, activation='softmax'))
#
#     # compile model using accuracy to measure model performance
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
#     # train the model
#     # model.fit(X_train[:5000], y_train[:5000], validation_data=(X_test[:1000], y_test[:1000]), epochs=3)
#     model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)
#
#     # save model
#     model.save("my_model1.model")
#
#     # evaluate model
#     val_loss, val_acc = model.evaluate(X_test, y_test)
#     print("Val loss " + str(val_loss))
#     print("Val acc " + str(val_acc))
#
#     return model
#
#
# # def predict_model():
# #     X_train, X_test, y_train, y_test = prepare_data()
# #
# #     model = prepare_model()
# #     pred = model.predict(X_test[:1])
# #     result = np.where(pred == np.amax(pred))
# #
# #     return result[0]
#
#
# def predict_my_data():
#     #
#     X_train, X_test, y_train, y_test = prepare_data()
#
#     # plot mnist image
#     ctm.plot_image()
#
#     # cnn
#     model = None
#     model = prepare_model()
#
#     # receive mnist number
#     my_number = ctm.array_to_export()
#     my_number = my_number.reshape(1, 28, 28, 1)
#
#     # save model
#     model.save("my_model1.model")
#     new_model = tf.keras.models.load_model("my_model1.model")
#
#     # predict my number
#     pred = new_model.predict(my_number[:1])
#
#     result = np.where(pred[0] == np.amax(pred[0]))
#
#     # # evaluate model
#     # val_loss, val_acc = new_model.evaluate(X_test, y_test)
#     # print("Val loss " + str(val_loss))
#     # print("Val acc " + str(val_acc))
#
#     # print(y_test)
#
#     return int(result[0])
#
# if __name__ == "__main__":
#     #result = predict_model()
#     #print(result)
#     model = prepare_model()

from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = None
model = Sequential()

# 1
# model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
# model.add(Conv2D(32, kernel_size=3, activation='relu'))

# 2
model.add(Conv2D(128, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, kernel_size=3, activation='relu'))

# 3
# model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
# model.add(Conv2D(16, kernel_size=3, activation='relu'))

# 4
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
# model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))

# 5
# model.add(Conv2D(64, kernel_size=(1, 1), activation='relu', input_shape=(28, 28, 1)))
# model.add(Conv2D(32, kernel_size=(1, 1), activation='relu'))

# 6
# model.add(Conv2D(128, kernel_size=(1, 1), activation='relu', input_shape=(28, 28, 1)))
# model.add(Conv2D(64, kernel_size=(1, 1), activation='relu'))

# 7
# model.add(Conv2D(32, kernel_size=(1, 1), activation='relu', input_shape=(28, 28, 1)))
# model.add(Conv2D(16, kernel_size=(1, 1), activation='relu'))

# 8
# model.add(Conv2D(64, kernel_size=1, activation='relu', input_shape=(28, 28, 1)))
# model.add(Conv2D(32, kernel_size=1, activation='relu'))
# model.add(Conv2D(16, kernel_size=1, activation='relu'))


model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)
# model.save("my_model8.model")


