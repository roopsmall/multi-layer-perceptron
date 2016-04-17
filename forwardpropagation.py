
import numpy as np
from numpy.random import rand


class ForwardPropagation:


	def __init__(self, shape, Phi=0):
                """define the shape and initialise the variables for each layer"""

                # make these available to all methods
		self.shape = np.array(shape)
                self.Phi = Phi
                self.W = []
                self.b = []

                # initialise logging variables
                self.log = np.array([0,0,0,0])

                # initialise the matrices for each MLP layer
                self._random_initialisation(shape)


	def _outit(self):
                """jobs to run to clean up at the end of a single iteration of forward propagation"""

		# reset the logging variables
		self.log = np.array([0,0,0,0])


	def _random_initialisation(self, shape):
                """initialise the MLP matrices with random values"""

		for i in range(len(shape)-1):
                        rows = shape[i+1]
                        cols = shape[i] + 1
			self.W.append(np.sqrt(12./(rows*cols))*np.matrix(rand(rows, cols)))
			self.b.append(np.sqrt(12./cols)*np.matrix(rand(cols)))


	def forward(self, X, function=):
                """forward propagtion of Multi-Layer-Perceptron"""
		pass


	def logging(self, tags):
                """return some statistics of the most recent forward propagation"""

		for tag in tags:
			if tag == "W": self.log[0] = 1
			elif tag == "b": self.log[1] = 1
			elif tag == "z": self.log[2] = 1
			elif tag == "G": self.log[3] = 1
