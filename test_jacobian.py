import theano
import theano.tensor as T
import numpy

def jacobian_mul_vector_r(y, x, W, v, x_val, W_val, v_val):
    J = theano.gradient.jacobian(y, x)
    JV = J.dot(v)
    f_JV = theano.function([x, W, v], JV)
    return f_JV(x_val, W_val, v_val)

def jacobian_mul_vector_r_flat(y, x, W, v, x_val, W_val, v_val):
    J = theano.gradient.jacobian(y, x)
    J_flat = T.flatten(J, J.ndim - 1) # The jacobian result on flattened matrix x
    JV = J_flat.dot(v.flatten())
    f_JV = theano.function([x, W, v], JV)
    return f_JV(x_val, W_val, v_val)

def jacobian_mul_vector_l(y, x, W, v, x_val, W_val, v_val):
    J = theano.gradient.jacobian(y, x)
    VJ = v.dot(J)
    f_VJ = theano.function([x, W, v], VJ)
    return f_VJ(x_val, W_val, v_val)

def jacobian_mul_vector_l_flat(y, x, W, v, x_val, W_val, v_val):
    J = theano.gradient.jacobian(y, x)
    J_flat = T.flatten(J, J.ndim - 1) # The jacobian result on flattened matrix x
    VJ = v.dot(J_flat)
    VJ_reshape = T.reshape(VJ, T.shape(x))
    f_VJ = theano.function([x, W, v], VJ_reshape)
    return f_VJ(x_val, W_val, v_val)

W = T.matrix('W')
V = T.matrix('V')
v  = T.vector('v')
x = T.vector('x')
y = T.dot(x, W)

x_val = numpy.array([1, 2])
W_val = numpy.array([[2, 3], [4, 5]])
v_val = numpy.array([5, 6])
V_val = numpy.array([[5, 6], [7, 8]])

# test vectors
"""
Jvr = T.Rop(y, x, v)
f_vr = theano.function([W, v, x], Jvr)
print "R result:"
print f_vr(W_val, v_val, x_val)
print jacobian_mul_vector_r(y, x, W, v, x_val, W_val, v_val)
print
print "L result:"
Jvl = T.Lop(y, x, v)
f_vl = theano.function([W, v], Jvl)
print f_vl(W_val, v_val)
print jacobian_mul_vector_l(y, x, W, v, x_val, W_val, v_val)
"""

# test matrix
JVr = T.Rop(y, W, V)
f_Vr = theano.function([W, V, x], JVr)
print "R result:"
print f_Vr(W_val, V_val, x_val)
print jacobian_mul_vector_r(y, W, x, V, W_val, x_val, V_val)
print jacobian_mul_vector_r_flat(y, W, x, V, W_val, x_val, V_val)
print
print "L result:"
Jvl = T.Lop(y, W, v)
f_vl = theano.function([x, v], Jvl)
print f_vl(x_val, v_val)
print jacobian_mul_vector_l(y, W, x, v, W_val, x_val, v_val)
print jacobian_mul_vector_l_flat(y, W, x, v, W_val, x_val, v_val)
