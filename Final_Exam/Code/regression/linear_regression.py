import scipy
import theano
import theano.tensor as T

# Define variables.
W = T.matrix('weights')
x = T.matrix('features')
z = T.matrix('targets')

# Define model. (Yes, no bias, I am lazy)
y = T.dot(x, W)

# Define loss function.
error = ((z - y)**2).sum()

# Build gradient expression.
d_error_wrt_W = T.grad(error, [W])

# Compile error function and its gradient.
f = theano.function([W, x, z], error)
f_prime = theano.function([W, x, z], d_error_wrt_W)

# Some constants.
steprate = 0.001
iterations = 1000

# Define data. This is is just a one dimensional problem.
features = scipy.array([[1], [2], [1.5]])
targets = scipy.array([[2], [4], [3]])
parameters = scipy.random.standard_normal((1,1))


for i in range(iterations):
    error = f(parameters, features, targets)
    errorgrad = f_prime(parameters, features, targets)
    parameters -= errorgrad * steprate
    print error


print parameters