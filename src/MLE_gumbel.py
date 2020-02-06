from __future__ import division
import numpy as np
import math
import matplotlib.pylab as plt
import sympy as sp
from numpy.linalg import inv
from time import sleep
import matplotlib.pyplot as plt
import statistics

class gumbelEstimation(object):

    def __init__(self):       
        self.gumbel_dist = []
        self.tolerance = 0.000000000000000001
        self.a = 2.3
        self.b = 4.0
        

    def pdf(self, a :float, b :float, x):
        pdf = (1/b) * math.exp(-(x - a)/b) * math.exp(-(math.exp(-(x - a)/b)))
        return pdf

    def randomGumbelDistGenerator(self, n):
        '''
        Function to generate random gumbel distribution
        '''
        gumbel_dist = np.random.gumbel(self.a, self.b, n)
        return gumbel_dist

    def newtonRaphsonOptimizer(self, gumbel_dist):
        '''
        Function to optimize the multivariate equations using newton raphson
        '''
        s = statistics.stdev(gumbel_dist)
        m = statistics.mean(gumbel_dist)

        #Setting initial values
        a = m - 0.4501 * s
        b = 0.7977 * s
    
        while True:
            # A Derivative
            a_derivate = (1/b) * (len(gumbel_dist) - sum((math.exp(-(x_i-a)/b) for x_i in gumbel_dist)))
            # B Derivative
            b_derivate = sum(((x_i - a)/b**2) for x_i in gumbel_dist) - len(gumbel_dist)/b - sum(((x_i - a)/b**2)*math.exp(-(x_i - a)/b) for x_i in gumbel_dist)
            # A Double Derivative
            a_derivate2 = -(1/b**2)*sum(math.exp((-(x_i - a)/b)) for x_i in gumbel_dist)
            # B Double Derivative
            b_derivate2 = len(gumbel_dist)/b**2 - (2/b**3) * sum((x_i - a) for x_i in gumbel_dist) + (2/b**3)*sum((x_i - a) * math.exp(-(x_i - a)/b) for x_i in gumbel_dist) - (1/b**4)*sum(((x_i - a)**2) * math.exp(-(x_i - a)/b) for x_i in gumbel_dist)
            # AB Derivative
            ab_derivate = -(len(gumbel_dist)/b**2) + (1/b**2) * sum(math.exp(-(x_i - a)/b) for x_i in gumbel_dist) - (1/b**3)*sum((x_i - a) * math.exp(-(x_i - a)/b) for x_i in gumbel_dist)
            print(a)
            f = np.matrix([[a_derivate], [b_derivate]], dtype = np.float64)
            theta = np.sum(f)
            h = np.matrix([[a_derivate2, ab_derivate], [ab_derivate, b_derivate2]], dtype = np.float64)
            h_inv = np.linalg.pinv(h)
            x_0 = np.matrix([[a], [b]], dtype=np.float64)
            x_1 = x_0 - (h_inv * f) 

            t = (x_0[0,0] - x_1[0,0]) * (x_0[0,0] - x_1[0,0]) + (x_0[1,0] - x_1[1,0]) * (x_0[1,0] - x_1[1,0])
   
            if t <= self.tolerance:
                break

            a = x_1[0,0] 
            b = x_1[1,0]
            sleep(0)

        return a,b


if __name__ == '__main__':
    gum = gumbelEstimation() 

    a_mean = []
    b_mean = []
    a_stdev = []
    b_stdev = []
    for i in range(3):
        #Outer loop for running it for n = 10,100,1000
        a_val = []
        b_val = []
        for j in range(10):
            #Inner loop for running it for 10 times each
            gdist = gum.randomGumbelDistGenerator(100*10**i)
            a, b = gum.newtonRaphsonOptimizer(gdist)
            a_val.append(a)
            b_val.append(b)
        #average values and finding std deviations
        a_mean.append(np.mean(a_val))
        b_mean.append(np.mean(b_val))
        a_stdev.append(np.std(a_val))
        b_stdev.append(np.std(b_val))
    #Printing all the generated data
    print(a_mean)
    print(b_mean)
    print(a_stdev)
    print(b_stdev)
        








