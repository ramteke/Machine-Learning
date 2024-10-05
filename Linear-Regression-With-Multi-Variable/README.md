# Linear-Regression-With-Multiple-Variable

You are selling your house and you want to know what a good market price would be. 

Predict the price of a house with 1650 square feet and 3 bedrooms

----------- input ----------- <br>
ex1data2.txt contains a training set of housing prices in Port- land, Oregon. 
The first column is the size of the house (in square feet), the second column is the number of bedrooms, and the third column is the price of the house.

α = 0.01;<br>
num_iters = 400;

Given predict the price of a house with 1650 square feet and 3 bedrooms ?


<b>How to Solve this ?</b><br>

<img src="Linear%20Regression%20-%20Multiple.png">

Normalize the data, i.e  normalized_value = current_value * mean / standard-deviation

Based on formula, calculate theta for every iteration:
  
  theta = theta - (alpha / m) * (X'*X*theta - X' * y);
  
Now, Predict for some value house with 1650 square feet and 3 bedrooms ?
  
  normalize_size =  1650 - mean(sq.ft) / std(sq.ft) <br>
  normalize_rooms = 3 - mean(rooms) / std(rooms)<br>
  
  Y = [ 1 , normalize_size,  normalize_rooms] * theta

  
 Reference: 
 http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex3/ex3.html
 https://www.coursera.org/learn/machine-learning
