#initialize alpha beta mu and sigma for number of distributions k, k = 2
#each k, run the em algorithm

from __future__ import division
import numpy as np
import math
import matplotlib.pylab as plt
import sympy as sp
import scipy.stats as ss
from numpy.linalg import inv
from time import sleep
import matplotlib.pyplot as plt
import statistics

class EM(object):
    tolerance = 0.00000000001   #setting a tolerance for both newtonRaphson and EM
    def newtonRaphsonOptimizer(self, a, b, gumbel_dist, weight_matrix):
            '''
            Newton Raphson Optimizer, modified to be able to multiply probabilities too.
            '''
            while True:
                a_derivatedash = sum((-weight_matrix[i]/b**2) * math.exp(-(gumbel_dist[i] - a)/b) for i in range(len(gumbel_dist))) 
                a_derivate_b_derivate = sum(-(p_i/b**2) for p_i in weight_matrix) + sum((1/b**2) * weight_matrix[i] * math.exp(-(gumbel_dist[i] - a)/b) for i in range(len(gumbel_dist))) - sum(((1/b**3) * weight_matrix[i] * ((gumbel_dist[i] - a)) * math.exp(-(gumbel_dist[i] - a)/b) for i in range(len(gumbel_dist))))
                a_derivate = sum((weight_matrix[i]/b) - (weight_matrix[i]/b) * math.exp(-(gumbel_dist[i] - a)/b) for i in range(len(gumbel_dist)))
                b_derivate = sum((-weight_matrix[i]/b) + weight_matrix[i]*((gumbel_dist[i] - a)/b**2) - weight_matrix[i]*((gumbel_dist[i] - a)/b**2)*math.exp(-(gumbel_dist[i] - a)/b) for i in range(len(gumbel_dist)))
                b_derivatedash = sum((weight_matrix[i]/b**2) - ((2*weight_matrix[i])/b**3) * (gumbel_dist[i] - a) + ((2*weight_matrix[i])/b**3) * (gumbel_dist[i] - a) * math.exp(-(gumbel_dist[i] - a)/b) - ((weight_matrix[i])/b**4) * (gumbel_dist[i] - a)**2 * math.exp(-(gumbel_dist[i] - a)/b)  for i in range(len(gumbel_dist)))
                
                f = np.matrix([[a_derivate], [b_derivate]], dtype = np.float64)
                theta = np.sum(f)
                h = np.matrix([[a_derivatedash, a_derivate_b_derivate], [a_derivate_b_derivate, b_derivatedash]], dtype = np.float64)
                h_inv = np.linalg.pinv(h)
                x_0 = np.matrix([[a], [b]], dtype=np.float64)
                x_1 = x_0 - (h_inv * f) 

                t = (x_0[0,0] - x_1[0,0]) * (x_0[0,0] - x_1[0,0]) + (x_0[1,0] - x_1[1,0]) * (x_0[1,0] - x_1[1,0])
    
                if t <= self.tolerance:
                    #print(t)
                    #print("breaking")
                    break

                a = x_1[0,0] 
                b = x_1[1,0]
                #print(a,b)
                #print(t)
                sleep(0)

            return a,b

    def solveGaussian(self, dataset):
        mu = np.mean(dataset)
        sigma = statistics.stdev(dataset)
        #print(mu,sigma)
        return mu,sigma

    def pdf_gumbel(self, a, b, x):
        return (1/b)*(math.exp(-(x-a)/b))*(math.exp(-(math.exp(-(x-a)/b))))

    def pdf_gaussian(self, mu, sigma, x):
        return (1/(math.sqrt(2*math.pi*(sigma**2)))) * math.exp(-((x-mu)**2))/(2*sigma**2)

    def em_algorithm(self, dataset):
        s = statistics.stdev(dataset)
        m = statistics.mean(dataset)
        weight_gaussian = 0.4
        weights_gumbel = 0.6
        q75, q25 = np.percentile(dataset, [75 ,25])
        a = m - 0.4501 * s
        b = 0.7977 * s
        # a = statistics.median(dataset)
        # b = q75 - q25
        mu = statistics.median(dataset)
        sigma = q75 - q25
        t = 0
        while t < 300:
            prob_gauss = []
            prob_gum = []
            for i in range(len(dataset)):
                posterior_gauss = ss.norm(mu, sigma).pdf(dataset[i])
                posterior_gumb = self.pdf_gumbel(a, b, dataset[i])
                point_prob_gaussian = (weight_gaussian * posterior_gauss)/(weight_gaussian* posterior_gauss + posterior_gumb*weights_gumbel)
                #point_prob_gumbel = 1 - point_prob_gaussian
                point_prob_gumbel = ((posterior_gumb*weights_gumbel)/(weight_gaussian* posterior_gauss + posterior_gumb*weights_gumbel))
                prob_gauss.append(point_prob_gaussian)
                prob_gum.append(point_prob_gumbel)

            weight_gaussian2 = sum(p_i for p_i in prob_gauss)/len(dataset) 
            weight_gumbel2 = sum(p_i for p_i in prob_gum)/len(dataset)
            mu2 = sum(prob_gauss[i]*dataset[i] for i in range(len(dataset)))/sum(p_i for p_i in prob_gauss)
            sigma2 = math.sqrt(sum((dataset[i] - mu)**2*prob_gauss[i] for i in range(len(dataset)))/sum(p_i for p_i in prob_gauss))
            #sigma = (math.sqrt(((sum(dataset[i] - mu)**2)*prob_gauss[i]) for i in range(len(dataset)) / sum(p_i for p_i in prob_gauss)))

            a2, b2 = self.newtonRaphsonOptimizer(a, b, dataset, prob_gum)
            k = (a2 - a)**2 + (b2 - b)**2 + (mu2 - mu)**2 + (sigma2 - sigma)**2 + (weight_gaussian2 - weight_gaussian)**2 + (weight_gumbel2 -weights_gumbel)**2
            if  k < self.tolerance:
                print(k)
                print("breaking")
                break
            a = a2
            b = b2
            mu = mu2
            sigma = sigma2
            weights_gumbel = weight_gumbel2
            weight_gaussian = weight_gaussian2
            t+=1

        return a,b,mu,sigma,weights_gumbel,weight_gaussian

    def run(self,):
        '''
        Function that generates the mixed distributions and calls the EM function
        '''
        #Below are the values from which the distribution is created
        #a     = -4.0
        #b     = 2.1
        #mu    = 3.9
        #sigma = 3.5
        #w1    = 0.6
        #w2    = 0.5
        a_mean = []
        b_mean = []
        mu_mean = []
        sigma_mean = []
        w1_mean = []
        w2_mean = []
        a_stdev = []
        b_stdev = []
        mu_stdev = []
        sigma_stdev = []
        w1_stdev = []
        w2_stdev = []
        for i in range(2):
            a_val = []
            b_val = []
            mu_val = []
            sigma_val = []
            w1_val = []
            w2_val = []
            for j in range(10):
                print("loop" + str(j+1) + "," + "n=" +str(100*10**i))
                distributions = [
                    {"type": np.random.gumbel, "kwargs": {"loc": -4.0, "scale": 2.1}},
                    {"type": np.random.normal, "kwargs": {"loc": 3.9, "scale": 3.5}},
                ]
                coefficients = np.array([0.6, 0.4])
                coefficients /= coefficients.sum()      # in case these did not add up to 1
                sample_size = 100*10**(i)

                num_distr = len(distributions)
                data = np.zeros((sample_size, num_distr))
                for idx, distr in enumerate(distributions):
                    data[:, idx] = distr["type"](size=(sample_size,), **distr["kwargs"])
                    plt.hist(data[:, idx], bins=100, density=True)
        
                random_idx = np.random.choice(np.arange(num_distr), size=(sample_size,), p=coefficients)
                sample = data[np.arange(sample_size), random_idx]
                a,b,mu,sigma,w1,w2 = self.em_algorithm(sample)
                a_val.append(a)
                b_val.append(b)
                mu_val.append(mu)
                sigma_val.append(sigma)
                w1_val.append(w1)
                w2_val.append(w2)
            
            a_mean.append(np.mean(a_val))
            b_mean.append(np.mean(b_val))
            mu_mean.append(np.mean(mu_val))
            sigma_mean.append(np.mean(sigma_val))
            w1_mean.append(np.mean(w1_val))
            w2_mean.append(np.mean(w2_val))

            a_stdev.append(np.std(a_val))
            b_stdev.append(np.std(b_val))
            mu_stdev.append(np.std(mu_val))
            sigma_stdev.append(np.std(sigma_val))
            w1_stdev.append(np.std(w1_val))
            w2_stdev.append(np.std(w2_val))
        print("values of a_mean", str(a_mean)) 
        print("values of b_mean", str(b_mean))  
        print("values of mu_mean", str(mu_mean))  
        print("values of sigma_mean", str(sigma_mean))  
        print("values of w1_mean", str(w1_mean))  
        print("values of w2_mean", str(w2_mean))  

        print("values of a_stdev", str(a_stdev)) 
        print("values of b_stdev", str(b_stdev))  
        print("values of mu_stdev", str(mu_stdev))  
        print("values of sigma_stdev", str(sigma_stdev))  
        print("values of w1_stdev", str(w1_stdev))  
        print("values of w2_stdev", str(w2_stdev))  

if __name__ == '__main__':
    EMdemo = EM()
    EMdemo.run()