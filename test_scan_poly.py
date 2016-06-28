# -*- encoding: utf-8 -*-
import theano
import theano.tensor as T
import numpy

coefficients = T.vector("coefficients")
x = T.scalar("x")

max_coefficients_supported = 10000

components, updates = theano.scan(
                                  # lambda函数的参数顺序为sequences, prior results, non-sequences, 上述任何一个为集合则展开
                                  fn=lambda coefficient, power, free_variable: coefficient * (free_variable ** power),
                                  outputs_info=None,
                                  sequences=[coefficients, T.arange(max_coefficients_supported)], # 自动截断于最短的sequence
                                  non_sequences=x)
polynomial = components.sum()

calculate_polynomial = theano.function(inputs=[coefficients, x], outputs=polynomial)

# Test
test_coefficients = numpy.array([1, 0, 2])
test_value = 3
print(calculate_polynomial(test_coefficients, test_value))
print(1.0 * (3 ** 0) + 0.0 * (3 ** 1) + 2.0 * (3 ** 2))
