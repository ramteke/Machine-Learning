__author__ = 'Venkatesh Ramteke'
import numpy

def readMatrix(fileName):

    f = open(fileName, "r")

    X = numpy.zeros(shape=(0,2))
    Y = numpy.zeros(shape=(0,1))
    for line in f:
        line = line.replace("\n","");
        x,y = line.split(",");
        X = numpy.vstack([X, [1, float(x)]])
        Y = numpy.vstack([Y, [float(y)]])

    f.close()

    return X,Y


def computeCost(X, Y, theta):
    ## Formula: J(Θ) = 1.0 / 2m * SUM [ Xi * theta - Yi ]^2

    m = len(Y);

    var1 = X.dot(theta)
    var2 = var1 - Y
    var3 = numpy.square(var2)

    allSquareSum = numpy.sum(var3)

    J = 1.0/(2.0*m) * allSquareSum;

    return J;


# Refer: https://stackoverflow.com/questions/23984925/gradient-descent-in-matlab
# for explanation of using X'
def gradientDescent(X, Y, theta, alpha, iterations):
    m = len(Y)

    for i in range(0, iterations):
        ## theta = theta - ( alpha / m) * SUM ( X' * (X*theta - Y ) )

        var1 = X.dot(theta) - Y
        var2 = numpy.transpose(X).dot(var1)

        theta = theta - (alpha / m) * var2;

        J = computeCost(X, Y, theta);  ## Optional, Validation Step, value of J(θ) should never increase
        print("J: " , J); ## testing:

    return theta;



### We start here

# Create matrix from file. Add x0 as all 1s to X so that Θ0 can be used as feature
X,Y = readMatrix('ex1data1.txt')

# Create theta
theta = numpy.zeros(shape=(2,1))

iterations = 1500;
alpha = 0.01;

theta = gradientDescent(X, Y, theta, alpha, iterations);

## now that we have theta...we can start predicting
## Given population of 70000 or 3.5 (X) in input scale...what is profit
var1 = numpy.matrix('1 7')

predict_y = var1.dot(theta)

print("For population of 70K, profit is:  " , predict_y)

## Done :)