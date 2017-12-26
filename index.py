import numpy as np

# X = (hours studying, hours sleeping), y = score on test, xPredicted = 4 hours studying & 8 hours sleeping (input data for prediction)
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)
xPredicted = np.array(([4, 8]), dtype=float)

# scale units
X = X/np.amax(X, axis=0) # maximum of X array
xPredicted = xPredicted/np.amax(xPredicted, axis=0) # maximum of xPredicted (our input data for the prediction)
y = y/100 # max test score is 100

class Neural_Network(object):
  def __init__(self):
    #parameters
    self.inputSize = 2
    self.outputSize = 1
    self.hiddenSize = 3

    #weights
    # self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (3x2) weight matrix from input to hidden layer
    # self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix from hidden to output layer

    # self.W1 = np.array(([
    #   0.5, 
    #   0.5, 
    #   0.5
    # ], [ 
    #   0.5, 
    #   0.5, 
    #   0.5
    # ]), dtype=float)

    # self.W2 = np.array(([
    #   [0.5], 
    #   [0.5], 
    #   [0.5]
    # ]), dtype=float)

    self.W1 = np.array(([
      0.554776535164,
      0.554776535164,
      0.554776535164
    ], [ 
      0.679216664472,
      0.679216664472,
      0.679216664472
    ]), dtype=float)

    self.W2 = np.array(([
      [ 0.993301448355 ],
      [ 0.993301448355 ],
      [ 0.993301448355 ]
    ]), dtype=float)

  def forward(self, X):
    #forward propagation through our network
    self.z = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights

    self.z2 = self.sigmoid(self.z) # activation function    
    self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
    o = self.sigmoid(self.z3) # final activation function

    return o

  def sigmoid(self, s):
    # activation function
    sigmoidResult = 1/(1+np.exp(-s))
    return sigmoidResult

  def sigmoidPrime(self, s):
    #derivative of sigmoid
    sigmoidPrimeResult = s * (1 - s) 
    return sigmoidPrimeResult

  def backward(self, X, y, o):
    # backward propgate through the network
    self.o_error = y - o # error in output
    self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error

    self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
    
    z2sigmoidPrime = self.sigmoidPrime(self.z2)
    self.z2_delta = self.z2_error * z2sigmoidPrime # applying derivative of sigmoid to z2 error

    a = X.T
    b = a.dot(self.z2_delta)
    self.W1 += b # adjusting first set (input --> hidden) weights

    a = self.z2.T
    b = a.dot(self.o_delta)
    self.W2 += b # adjusting second set (hidden --> output) weights

  def train(self, X, y):
    o = self.forward(X)
    self.backward(X, y, o)

  def saveWeights(self):
    np.savetxt("w1py.txt", self.W1, fmt="%s")
    np.savetxt("w2py.txt", self.W2, fmt="%s")

  def predict(self):
    # print("Input (scaled): \n" + str(xPredicted))
    print("Final Output: " + str(self.forward(xPredicted)))

def run():
  NN = Neural_Network()
  for i in range(0, 1000): # trains the NN 1,000 times
    forward = NN.forward(X)
    difference = y - forward
    square = np.square(difference)
    loss = np.mean(square)

    # print("\n############################\n")
    # print("Input (scaled):" + str(X))
    # print("Actual Output:" + str(y))
    # print("Predicted Output:" + str(forward))
    # print("Loss >>> " + str(loss)) # mean sum squared loss
    # print("############################")
    # print("\n")

    NN.train(X, y)

  NN.saveWeights()
  NN.predict()

for j in range(0, 10):
  run()