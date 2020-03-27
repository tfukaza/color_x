from pyquery import PyQuery as pq
import colorsys as cs

from sklearn.svm import OneClassSVM 
from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

# https://towardsdatascience.com/outlier-detection-with-one-class-svms-5403a1a1878c
# https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html

N = 12

def main():

    # data points
    X = np.zeros((1,N+1))


    # open the html file of colorhunt.io
    f = open("index.html", "r") 
    # Parse as DOM
    dom = pq(f.read())

    # get all color palettes
    palette_div = dom('.palette')

    # create a tuple for each palette
    for palette in palette_div:

        # get the colors in each palette, and convert it into text
        colors = pq(palette)('span').text()

        # for each color, convert it into a tuple form expressed as a float in [0,1]
        # and generate a vector of 4 colors, 3 features each (ie N = 12)
        color_vector = text_to_vector(colors, 1)

        if len(color_vector) == 0:
            continue 

        X = np.append(X, color_vector, axis = 0)
    
    X = X[1:, :]
   
    

    #X_train_good = X_train[X_train[:, 8] == 1]
    #X_train_bad = X_train[X_train[:, 8] == -1]
    # create several artificially bad color palettes

    # https://www.color-hex.com/color-palette/6519
    ugly_colors = [
        "#FFE702 #FF0010 #00EB1E #C9FF00",
        "#ff006f #e5ff00 #00ff26 #004cff",
        "#3B59ED #FFBA44 #ACBAFA #262938",
        "#C960FF #ED3266 #352A3A #DB99FD",
        "#FF6B6B #EEAEAE #3A2B2B #1BC3A9"


    ]

    # for color in ugly_colors:
    #     X = np.append(X, text_to_vector(color, -1), axis = 0)

    X = shuffle(X)

    # split the data into train and test set
    X_train, X_test = train_test_split(X, test_size=0.2)

    for color in ugly_colors:
        X_test = np.append(X_test, text_to_vector(color, -1), axis = 0)
    X_test = shuffle(X_test)

    #print(len(X))
    # according to doc..
    # An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors. Should be in the interval (0, 1]. By default 0.5 will be taken.
    outlier = len(ugly_colors) / len(X) * 2

    # declare classifier
    svm = OneClassSVM(kernel='rbf', nu = outlier, gamma = 0.00009)

    svm.fit(X_train[:, 0:N])

    y_true = X_test[:, N]
    y_pred = svm.predict(X_test[:, 0:N]) 
    print(confusion_matrix(y_true, y_pred))

    # plt.subplot(321)
    # x = X_test[:, 0]
    # y = X_test[:, 2]
    # plt.scatter(x,y,alpha=0.7, c=X_test[:, N])
    
    # plt.subplot(322)
    # y_pred = svm.predict(X_test[:, 0:N])
    # plt.scatter(x,y,alpha=0.7, c=(y_pred + 1) // 2)

    # plt.subplot(323)
    # x = X_test[:, 1]
    # y = X_test[:, 3]
    # plt.scatter(x,y,alpha=0.7, c=X_test[:, N])
    
    # plt.subplot(324)
    # y_pred = svm.predict(X_test[:, 0:N])
    # plt.scatter(x,y,alpha=0.7, c=(y_pred + 1) // 2)

    # plt.subplot(325)
    # x = X_test[:, 4]
    # y = X_test[:, 6]
    # plt.scatter(x,y,alpha=0.7, c=X_test[:, N])
    
    # plt.subplot(326)
    # y_pred = svm.predict(X_test[:, 0:N])
    # plt.scatter(x,y,alpha=0.7, c=(y_pred + 1) // 2)

    # plt.show()




def text_to_vector(colors, label):

    color_tuple= np.zeros((1,N+1))

    for i, hex_val in enumerate(colors.split(' ')):
        # convert to float tuple (R, G, B)
        c = hex_to_tuple(hex_val)
        # check for invalid data
        if len(c) == 0:
            return ()
        
        # convert to HLS
        hls_tuple = cs.rgb_to_hls(c[0], c[1], c[2])

        # only use hue and saturation
        color_tuple[0][3 * i] = hls_tuple[0]
        color_tuple[0][3 * i + 1] = hls_tuple[1] 
        color_tuple[0][3 * i + 2] = hls_tuple[2] 

    color_tuple[0][N] = label

    return color_tuple
    



def hex_to_tuple(val):

    if val == '':
        return ()

    R = int(val[1:3], 16) / float(255)
    G = int(val[3:5], 16) / float(255)
    B = int(val[5:7], 16) / float(255)

    return (R, G, B)



if __name__ == "__main__":
    main() 