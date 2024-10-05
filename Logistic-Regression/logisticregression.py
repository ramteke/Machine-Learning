__author__ = 'Venkatesh Ramteke'


import numpy
import scipy.optimize as op

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


def sigmoid(z):
    return 1 / (1 + numpy.exp(-z));



def CostFunc(theta, X, y):
    m = len(y)
    # J = sum ( (-y' * (log(sigmoid(X*theta)))) - ((1 - y)' * log(1 - sigmoid(X*theta))) ) / m;

    var1 = numpy.log(sigmoid(X.dot(theta)))
    var2 = numpy.transpose(y).dot(-1).dot(var1)

    var3 = numpy.transpose(1 - y);
    var4 = numpy.log(1 - sigmoid(X.dot(theta)))
    var5 = var3.dot(var4)

    J = sum( var2 - var5) / m

    return J

def Gradient(theta,x,y):
    m, n = x.shape
    theta = theta.reshape((n, 1));
    y = y.reshape((m, 1))
    sigmoid_x_theta = sigmoid(x.dot(theta));
    grad = ((x.T).dot(sigmoid_x_theta - y)) / m;
    return grad.flatten();


### We start here

# Create matrix from file. Add x0 as all 1s to X so that Θ0 can be used as feature
X,y = readMatrix('ex2data1.txt')

# Initialize fitting parameters
m, n = X.shape;
initial_theta = numpy.zeros(n);

# compute cost and gradiant
#cost = costFunction(initial_theta, X, y);


## Calculate optimal theta value
Result = op.fmin_tnc(func=CostFunc, x0=initial_theta, fprime=Gradient, args=(X, y));
optimal_theta = Result[0];


## Now lets predict some values
## For a student with scores 45 and 85, we predict an admission

## Create input matroix
input_matrix = numpy.zeros(shape=(0,3))
input_matrix = numpy.vstack([input_matrix, [1, 45, 85]])


prob = sigmoid(input_matrix.dot(optimal_theta)) >= 0.5;
print("Probablity of admission is: " , prob)



