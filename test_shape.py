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

    x_mat = T.matrix()
    x_mat_2_t3 = T.reshape(x_mat, (2, 2, 2))
    x_mat_2_vec = T.reshape(x_mat, (8,))
    reshape_f = theano.function([x_mat], [x_mat_2_t3, x_mat_2_vec])
    """
    t3_shape = T.lvector()
    vec_shape = T.lvector()
    x_mat_2_t3 = T.reshape(x_mat, t3_shape, 3)
    x_mat_2_vec = T.reshape(x_mat, vec_shape, 1)
    reshape_f = theano.function([x_mat, t3_shape, vec_shape], [x_mat_2_t3, x_mat_2_vec])
    """
    mat_2_t3_val, mat_2_vec_val = reshape_f(flat_mat_val)
    print 'reshape 2-d array to 3-d array:'
    print mat_2_t3_val
    print 'reshape 2-d array to 1-d array:'
    print mat_2_vec_val

if __name__ == '__main__':
    test_shape() 
