import pandas as pd
import numpy as np
from keras.models  import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import train_test_split

#%%

# data preprocessing

train_df = pd.read_csv('./data/fashion-mnist_train.csv')
test_df = pd.read_csv('./data/fashion-mnist_test.csv')

x_train = np.array(train_df, dtype='float32')[:, 1:] / 255 # normalize pixels
labels = np.array(train_df)[:, 0]
y_train = np.zeros((len(train_df), 10))

# generates one hot vectors from the training dataset
# for example, for dresses (y = 3), one hot vector = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train[np.arange(len(train_df)), labels] = 1 

x_test = np.array(test_df, dtype='float32')[:, 1:] / 255 # normalize pixels
labels = np.array(test_df)[:, 0]
y_test = np.zeros((len(test_df), 10))
y_test[np.arange(len(test_df)), labels] = 1 # one hot encoding on the test set labels

# 80/20 training/val split
x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=1)

# reshape (784,) vectors to (28, 28, 1) for model input
im_shape = (28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], *im_shape)
x_val = x_val.reshape(x_val.shape[0], *im_shape)
x_test = x_test.reshape(x_test.shape[0], *im_shape)

print('Shape of train set:', x_train.shape)
print('Shape of validation set:', x_train.shape)
print('Shape of test set:', x_train.shape)

#%%

# VGG model

model = Sequential([
            Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same', input_shape=im_shape),
            Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2,2), strides=2),
            Dropout(0.5),
            Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same'),
            Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same'),
            AveragePooling2D(pool_size=(2,2), strides=2),
            Dropout(0.5),
            Flatten(),
            Dense(256),
            LeakyReLU(),
            Dropout(0.5),
            Dense(128),
            LeakyReLU(),
            Dropout(0.5),
            Dense(10, activation='softmax')
        ])

model.summary()

#%%
model.compile(optimizer='Adam', loss='categorical_crossentropy',
               metrics=['accuracy'])

# validation
model.fit(x_train, y_train, epochs=50, batch_size=512, verbose=1)
score = model.evaluate(x_val, y_val, batch_size=512)
print(score) # accuracy = 93.8%



