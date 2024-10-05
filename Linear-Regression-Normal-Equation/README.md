# Linear Regression With Multiple Variable : Normal Equation

Note: This is easy way to compute and can be used when number of features are less.


You are selling your house and you want to know what a good market price would be.

Predict the price of a house with 1650 square feet and 3 bedrooms

----------- input ----------- <br>
ex1data2.txt contains a training set of housing prices in Port- land, Oregon.
The first column is the size of the house (in square feet), the second column is the number of bedrooms, and the third column is the price of the house.


Given predict the price of a house with 1650 square feet and 3 bedrooms ?


<b>How to Solve this ?</b><br>

Data Normalization is not required here !!! But the operation will be slow due to the cpu intensive formula. i.e<br>

theta = pinv(X' * X) * X'* y;


Now, Predict for some value house with 1650 square feet and 3 bedrooms ?


  Y = [ 1 , 1650, 3] * theta


 Reference:
 http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex3/ex3.html
 https://www.coursera.org/learn/machine-learning
