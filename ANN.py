import numpy as np
import scipy as optimize

#input data
X = np.array(([3,5],[5,1],[10,2]), dtype = float)
#output data
Y = np.array(([75],[82],[93]), dtype = float)

#scaling of the data
X = X/(np.amax(X, axis=0))
Y = Y/100.0

class Neural_Network(object):
	def __init__ (self):
		#hyperparameters
		self.inputLayerSize = 2
		self.hiddenLayerSize = 3
		self.outerLayerSize = 1

		self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
		self.W2 = np.random.randn(self.hiddenLayerSize,self.outerLayerSize)

	def forward (self, X):
		#propagates input through networks
		self.z2 = np.dot(X,self.W1)
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.a2, self.W2)
		yHat = self.sigmoid(self.z3)
		return yHat

	def costFunction(self, X, Y):
		#parabolic cost function
		#also allows to explot convexity of the function
		self.yHat = self.forward(X)
		J = 0.5*sum((Y-self.yHat)**2)
		return J

	def costFunctionPrime(self, X, Y):
		#computing derivative with respect to W1 and W2
		self.yHat = self.forward(X)
		delta3 = np.multiply(-(Y-self.yHat), self.sigmoidPrime(self.z3))
		dJdW2 = np.dot(self.a2.T, delta3)

		delta2 = np.dot(delta3, self.w2.T)*self.sigmoidPrime(self.z2)
		dJdW1 = np.dot(X.T, delta2)

		return dJdW1, dJdW2

	def sigmoid(self,z):
		return 1.0/(1.0+np.exp(-z))

	def sigmoidPrime(self,z):
		return self.sigmoid(z)*(1-self.sigmoid(z))

NN = Neural_Network()
y = NN.forward(X)

cost1 = NN.costFunction(X,Y)

print("X is:",X)
print("W1 is:",NN.W1)
print("W2 is:",NN.W2)
print("prediction is:",y)
print("cost1 is:",cost1)

#print(cost2)
#print(cost3)