##
# Regression and Classification programming exercises
##

#
#	In this exercise we will be taking a small data set and computing a linear function
#	that fits it, by hand.

#	the data set

import numpy as np

sleep = [5,6,7,8,10]
scores = [65,51,75,75,86]

def compute_regression(sleep,scores):
    #	First, compute the average amount of each list
    avg_sleep = np.average(sleep)
    avg_scores = np.average(scores)

    #	Then normalize the lists by subtracting the mean value from each entry
    normalized_sleep = [i-avg_sleep for i in sleep]
    normalized_scores = [i-avg_scores for i in scores]
#     normalized_sleep = ['{:.2f}'.format(i-avg_sleep) for i in sleep]
#     normalized_scores = ['{:.2f}'.format(i-avg_scores) for i in scores]
    print(normalized_sleep)
    print(normalized_scores)

    #	Compute the slope of the line by taking the sum over each student
    #	of the product of their normalized sleep times their normalized test score.
    #	Then divide this by the sum of squares of the normalized sleep times.
    slope = sum([x*y for x,y in zip(normalized_sleep,normalized_scores)]) / sum([np.square(y) for y in normalized_sleep])# = 0
    print(slope)
    #	Finally, We have a linear function of the form
    #	y - avg_y = slope * ( x - avg_x )
    #	Rewrite this function in the form
    #	y = m * x + b
    #	Then return the values m, b
    intercept = 0

    return slope,intercept
#    return m,b


if __name__=="__main__":
    m,b = compute_regression(sleep,scores)
    print("Your linear model is y={}*x+{}".format(m,b))