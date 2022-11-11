""""""  		  	   		  	  			  		 			     			  	 
"""  		  	   		  	  			  		 			     			  	 
template for generating data to fool learners (c) 2016 Tucker Balch  		  	   		  	  			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  	  			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		  	  			  		 			     			  	 
All Rights Reserved  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
Template code for CS 4646/7646  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  	  			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		  	  			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		  	  			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		  	  			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		  	  			  		 			     			  	 
or edited.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		  	  			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		  	  			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  	  			  		 			     			  	 
GT honor code violation.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
-----do not edit anything above this line---  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
Student Name: Rimon Mikhael (replace with your name)  		  	   		  	  			  		 			     			  	 
GT User ID: rmikhael3   (replace with your User ID)  		  	   		  	  			  		 			     			  	 
GT ID: 903737444 (replace with your GT ID)  		  	   		  	  			  		 			     			  	 
"""

import math
import numpy as np


# this function should return a dataset (X and Y) that will work
# better for linear regression than decision trees
def best_4_lin_reg(seed=1489683273):
    np.random.seed(seed)
    row = np.random.randint(100, 1000)
    col = np.random.randint(4, 10)
    x = np.random.random(size=(row, col))
    y = x[:, 1] + x[:, 2] ** 2 + x[:, 3] ** 3 + np.sin(x[:, 4])

    return x, y


def best_4_dt(seed=1489683273):
    np.random.seed(seed)
    row = np.random.randint(100, 1000)
    col = np.random.randint(2, 4)
    x = np.random.normal(size=(row, col))
    y = np.ones(row)

    for i in range(row): y[i] = 1 if (x[i, 1] > 0) else -1
    return x, y


def author():
    return "rmikhael3"  # Change this to your user ID


if __name__ == "__main__":
    print("they call me Tim.")

