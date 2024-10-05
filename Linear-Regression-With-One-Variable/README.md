# Linear regression with one variable (Gradient Decent)


Implement linear regression with one variable to predict profits for a food truck.
Suppose you are the CEO of a restaurant franchise and are considering different cities for opening a new outlet.
The chain already has trucks in various cities and you have data for profits and populations from the cities.

You would like to use this data to help you select which city to expand to next.

---------- Input -------<br>
The file ex1data1.txt contains the dataset for our linear regression problem.
The first column is the population of a city and the second column is the profit of a food truck in that city.
A negative value for profit indicates a loss.

α = 0.01<br>
iteration = 1500

Based on existing data, for city with population 70000, what is possible profit value ?


<b><font size=11>How to solve this ?</font></b>

To solve this we need below formula:

<img src='Gradient%20Descent%20Algo.png' height=200>

The α in the formula refers to learning rate. Go with too low value and we might reach optimal solution very slowly. With very high value, we might overshoot the optimal.

The 'repeat until' refers to number of times this formula should be executed.

        theta = theta - ( alpha / m) * SUM ( X' * (X*theta - Y ) )    
        
        WHAT !!!, refer: https://stackoverflow.com/questions/23984925/gradient-descent-in-matlab     


Now that theta is available, Try predicting Y value for some X

ex. for X = 7,   Y = [ 1 , 7 ] * theta


Reference:  https://www.coursera.org/learn/machine-learning
