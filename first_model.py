import keras
from keras.layers import Dense, Flatten, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Lambda, ELU
from keras.models import Sequential, load_model
import numpy as np
import pandas as pd
from OpenFiles import OpenFiles
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

np.random.seed(123)
batch_size = 30
num_classes = 2
epochs = 20

def main():
    path = "/Users/Joel/Desktop/robotics research/Obstacles"
    path2 = "/Users/Joel/Desktop/robotics research/Open"

    obstacle_data = OpenFiles(path).get_images()
    open_data_org = OpenFiles(path2).get_images()

    first = []
    sec = []

    for i in obstacle_data:
        first.append(i)
    for i in open_data_org:
        sec.append(i)

    # We will assign class 1 to open data
    # and class 2 to obstacle data

    label_obstacle_data  = [2 for i in range(len(obstacle_data))]
    label_open_data  = [1 for i in range(len(open_data_org))]

    label1 = pd.Series(label_obstacle_data)
    label2 = pd.Series(label_open_data)

    obstacle = pd.Series(first)
    open_data = pd.Series(sec)

    label2 = pd.Series(label_open_data)

    df1 = pd.DataFrame({'images': obstacle, 'label':label1})
    df2 = pd.DataFrame({'images': open_data, 'label':label2})

    frames = [df1, df2]
    dataset = pd.concat(frames)

    '''print(len(final_dataset))
    print(type(final_dataset.iloc[0,1]))
    print(final_dataset.iloc[0,0])'''
    #print(len(open_dict))

    targets = np.array(list(dataset.iloc[:,1]))
    temp = list(dataset.iloc[:,0])
    final_images = [np.array(i) for i in temp]

    final_images = np.array(final_images)
    print(targets)
    print(type(open_data_org))

    final_dataset = obstacle_data + open_data_org
    final_dataset = np.array(final_dataset)
    print(len(final_dataset))
    print(final_images.shape)
    #print([2 for i in range(len(obstacle_data))])
    '''targets = [2 for i in range(len(obstacle_data))] + [1 for i in range(len(open_data))]

    print(targets)
    #222 total
    #199 train
    '''

    #print(targets.shape)
    X_train, x_test, Y_train, y_test = train_test_split(final_images,
            targets, test_size=0.20)


    X_train = X_train.reshape(X_train.shape[0], 256,256,3)
    x_test = x_test.reshape(x_test.shape[0],256,256,3)
    print(X_train.shape)
    Y_train = to_categorical(Y_train, num_classes=3)
    y_test = to_categorical(y_test, num_classes=3)

    print("past preprocessing")
    print(Y_train.shape)
    print(Y_train)
    ### Model generation


    model = Sequential()
    model.add(Lambda(lambda x:x/127.5 -1, input_shape=(256,256,3)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    # 5x5 convolution (kernel) with 2x2 stride over 32 output filters
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    # 5x5 convolution (kernel) with 2x2 stride over 64 output filters
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    # Flatten the input to the next layer
    model.add(Flatten())
    # Apply dropout to reduce overfitting
    model.add(Dropout(.2))
    model.add(ELU())
    # Fully connected layer
    model.add(Dense(512))
    # More dropout
    model.add(Dropout(.5))
    model.add(ELU())
    # Fully connected layer with one output dimension (representing the speed).
    model.add(Dense(3))

    model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

    print("start traning")
    model.fit(X_train, Y_train, batch_size = 20, nb_epoch=100, verbose=1)
    score = model.evaluate(x_test, y_test, verbose=0)
    print(score)

    model.save('first_model2.h5')
    print("model has been saved")

main()
