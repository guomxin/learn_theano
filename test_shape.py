import theano
import theano.tensor as T
import numpy

tensor3_val = numpy.array([[[1, 2], [3, 4]],
                           [[5, 6], [7,8]]])

def test_shape():
    x = T.tensor3()
    x_flat_2_mat = T.flatten(x, 2)
    x_flat_2_vec = T.flatten(x, 1)
    flat_f = theano.function([x], [x_flat_2_mat, x_flat_2_vec])
    flat_mat_val, flat_vec_val = flat_f(tensor3_val)
    print 'flatten to 2-d array:'
    print flat_mat_val
    print 'flatten to 1-d array:'
    print flat_vec_val

    x_mat_2_t3 = T.reshape(x_flat_2_mat, T.shape(x))
    x_mat_2_vec = T.reshape(x_flat_2_mat, T.shape(x_flat_2_vec))
    reshape_f = theano.function([x], [x_mat_2_t3, x_mat_2_vec])
    mat_2_t3_val, mat_2_vec_val = reshape_f(tensor3_val)
    print 'reshape 2-d array to 3-d array:'
    print mat_2_t3_val
    print 'reshape 2-d array to 1-d array:'
    print mat_2_vec_val

if __name__ == '__main__':
    test_shape() 
