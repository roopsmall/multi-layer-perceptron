from forwardpropagation import *

# define shape of MLP
FP = ForwardPropagation([2, 3, 3, 2])

# set input vector
x = [2,2]

# print result
print FP.forward(x)
