__author__ = 'Venkatesh Ramteke'

import numpy

def readMatrix(fileName):

    f = open(fileName, "r")

    X = numpy.zeros(shape=(0,3))
    Y = numpy.zeros(shape=(0,1))
    for line in f:
        line = line.replace("\n","");
        x,y,z = line.split(",");
        X = numpy.vstack([X, [1, float(x), float(y)]])
        Y = numpy.vstack([Y, [float(z)]])

    f.close()

    return X,Y


def dataNormalization(X):
    mean_1 = numpy.mean([row[1] for row in X])
    mean_2 = numpy.mean([row[2] for row in X])

    std_1 =  numpy.std([row[1] for row in X], axis=0)
    std_2 = numpy.std([row[2] for row in X], axis=0)

    X_norm = numpy.zeros(shape=(0, 3))

    for i in range(0, len(X)):
        row = X[i]
        col1 =  (row[1] - mean_1) / std_1
        col2 = (row[2] - mean_2) / std_2
        X_norm = numpy.vstack([X_norm, [1, col1, col2]])

    return X_norm, mean_1, mean_2, std_1, std_2


def computeCost(X, Y, theta):
    ## Formula: J(Θ) = 1.0 / 2m * SUM [ Xi * theta - Yi ]^2

    m = len(Y);

    var1 = X.dot(theta)
    var2 = var1 - Y
    var3 = numpy.square(var2)

    allSquareSum = numpy.sum(var3)

    J = 1.0/(2.0*m) * allSquareSum;

    return J;


def gradientDescent(X, Y, theta, alpha, iterations):
    m = len(Y)

    for i in range(0, iterations):
        #theta = theta - (alpha / (m) * (X'*X*theta - X' * y);

        var1 = X.dot(theta)
        var2 = numpy.transpose(X).dot(var1)
        var3 = var2 - numpy.transpose(X).dot(Y)

        theta =  theta - (alpha/ m) * var3

        J_cost = computeCost(X, Y, theta);  ## Optional, Validation Step, value of J(θ) should never increase
        print("J: " , J_cost); ## testing:

    return theta;




### We start here

# Create matrix from file. Add x0 as all 1s to X so that Θ0 can be used as feature
X,Y = readMatrix('ex1data2.txt')
X,mean_1, mean_2, std_1, std_2 = dataNormalization(X)

alpha = 0.01;
num_iters = 400;

theta = numpy.zeros(shape=(3, 1))

theta = gradientDescent(X, Y, theta, alpha, num_iters)

## now that we have theta...we can start predicting
## Given predict the price of a house with 1650 square feet and 3 bedrooms

# Normalize 1650sq. , i.e
normalize_size = ( 1650 - mean_1 ) / std_1
normalize_rooms = ( 3 - mean_2) / std_2

var1 = numpy.matrix([1, normalize_size, normalize_rooms])

predict_y = var1.dot(theta)

print("Predicted price of a 1650 sq-ft, 3 br house is:   " , predict_y)

## Done :)