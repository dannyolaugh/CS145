import math
import numpy as np
#-------------------------------------------------------------------
def log(n):
    return math.log(n)
#-------------------------------------------------------------------
def exp(n):
    return math.exp(n)
#-------------------------------------------------------------------
class logistic:
    #******************************************************
    def __init__(self, parameters):
        self.parameters = parameters
    #******************************************************
    ########## Feel Free to Add Helper Functions ##########
    def x(self):
        return [[1, 60, 155], [1, 64, 135], [1, 73, 170]]

    def y(self):
        return [0, 1, 1]
    #******************************************************
    def log_likelihood(self):
        ll = 0.0
        ##################### Please Fill Missing Lines Here #####################

        beta = self.parameters
        x = self.x()
        y = self.y()

        y = [0, 1, 1]
        for i in range(0, len(y)):
            term = (x[i][0] * beta[0]) + (x[i][1] * beta[1]) + (x[i][2] * beta[2])
            ll += (y[i]*term) - np.log2(1 + exp(term))

        return ll
    #******************************************************
    def gradients(self):
        gradients = []
        ##################### Please Fill Missing Lines Here #####################
        x = self.x()
        beta = self.parameters
        y = self.y()
        gradient1 = 0
        gradient2 = 0
        gradient3 = 0

        for i in range(0, len(y)):
            term = exp(beta[0]*x[i][0] + beta[1]*x[i][1] + beta[2]*x[i][2])/(1 + exp(beta[0]*x[i][0] + beta[1]*x[i][1] + beta[2]*x[i][2]))
            gradient1 += x[i][0]*(y[i] - term)
            gradient2 += x[i][1]*(y[i] - term)
            gradient3 += x[i][2]*(y[i] - term)

        gradients.append(gradient1)
        gradients.append(gradient2)
        gradients.append(gradient3)

        return gradients
    #******************************************************
    def iterate(self):
        ##################### Please Fill Missing Lines Here #####################
        self.log_likelihood()
        self.gradients()
        return self.parameters
    #******************************************************
    def hessian(self):
        n = len(self.parameters)
        hessian = numpy.zeros((n, n))
        ##################### Please Fill Missing Lines Here #####################
        return hessian
#-------------------------------------------------------------------
parameters = [0.25, 0.25, 0.25]
##################### Please Fill Missing Lines Here #####################
## initialize parameters
l = logistic(parameters)
parameters = l.iterate()
l = logistic(parameters)
print (l.iterate())


