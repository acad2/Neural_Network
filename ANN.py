import numpy as np
from scipy import optimize

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

	def sigmoid(self,z):
		return 1.0/(1.0+np.exp(-z))

	def sigmoidPrime(self,z):
		return self.sigmoid(z)*(1-self.sigmoid(z))

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

		delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
		dJdW1 = np.dot(X.T, delta2)

		return dJdW1, dJdW2

	#numerical gradient checking

	#helper function to interact with other function
	def getParams(self):
		#rolling W1 and W2 rolled into a single vector
		params = np.concatenate((self.W1.ravel(),self.W2.ravel()))
		return params

	def setParams(self, params):
		#set W1 and W2 rolled into vectors
		W1_start = 0
		W1_end = self.inputLayerSize*self.hiddenLayerSize
		self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize, self.hiddenLayerSize))
		W2_end = W1_end + self.hiddenLayerSize*self.outerLayerSize
		self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outerLayerSize))

	def computeGradients(self, X,Y):
		#returns a single vector containg gradients with respect to different weights
		dJdW1, dJdW2 = self.costFunctionPrime(X, Y)
		return np.concatenate((dJdW1.ravel(),dJdW2.ravel()))
	
	def computeNumericalGradients(self, X, Y):
		paramsIntial = self.getParams()
		numgrad = np.zeros(paramsIntial.shape)
		perturb = np.zeros(paramsIntial.shape)
		e = 1e-4

		#I will perturb each weight one by one and calculate the
		#corresponding gradient
		#by the end we will get a vector having numericaly 
		#gradients calculated
		for p in range(len(paramsIntial)):
			perturb[p] = e
			self.setParams(paramsIntial+perturb)
			loss2 = self.costFunction(X,Y)

			self.setParams(paramsIntial-perturb)
			loss1 = self.costFunction(X,Y)

			numgrad[p] = (loss2-loss1)/(2*e)
			perturb[p] = 0

		self.setParams(paramsIntial)
		return numgrad

	def gradCheck(self, X, Y):
		grad = self.computeGradients(X,Y)
		numgrad = self.computeNumericalGradients(X,Y)
		return ((np.linalg.norm(grad-numgrad)/np.linalg.norm(grad+numgrad))<1e-8)

class trainer(object):
	def __init__(self, N):
		#local refernce to Neural_Network
		self.N = N

	#wrapper functions which returns the output which satisfies
	#the semantics of the minimize function
	def wrapper(self, params, X, Y):
		self.N.setParams(params)
		cost = self.N.costFunction(X, Y)
		grad = self.N.computeGradients(X,Y)
		return cost, grad

	def callbackF(self, params):
		self.N.setParams(params)
		self.J.append(self.N.costFunction(self.X, self.Y))

	def train(self, X, Y):
		self.X = X
		self.Y = Y

		self.J = []

		paramsI = self.N.getParams()

		options = {'maxiter': 200, 'disp':True}


		#minimize function from optimize requires a function
		#as parameter which accepts 2 vectors output and input
		#and returns the cost function and the gradients
		_res = optimize.minimize(self.wrapper, paramsI, jac = True, method='BFGS',\
		 						args = (X,Y), options = options, callback = self.callbackF)

		self.N.setParams(_res.x)
		self.optimizationResult = _res


NN = Neural_Network()
T = trainer(NN)
T.train(X,Y)
y = NN.forward(X)

cost1 = NN.costFunction(X,Y)
dJdW1,dJdW2 = NN.costFunctionPrime(X,Y)

#to demo the effect of moving along and opposite to the gradient
#slr = 3.0
#NN.W1 = NN.W1 + slr*dJdW1
#NN.W2 = NN.W2 + slr*dJdW2
#cost2 = NN.costFunction(X,Y)
#NN.W1 = NN.W1 - 2*slr*dJdW1
#NN.W2 = NN.W2 - 2*slr*dJdW2
#cost3 = NN.costFunction(X,Y)

gradient = NN.computeGradients(X,Y)
numGradient = NN.computeNumericalGradients(X,Y)
check = NN.gradCheck(X,Y)

#to check whether gradients are calculated correctly

print("X is:",X)
print("prediction is:",y)
print("cost1 is:",cost1)
#print("cost2 is:",cost2)
#print("cost3 is:",cost3)
print("gradient is:", gradient)
print("Numerical Gradient is:", numGradient)
print("check passed:", check)
print("W1 is:",NN.W1)
print("W2 is:",NN.W2)
