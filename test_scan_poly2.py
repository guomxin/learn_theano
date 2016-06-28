import theano
import theano.tensor as T
import numpy

coefficients = T.vector("coefficients")
x = T.scalar("x")

max_coefficients_supported = 10000

result, updates = theano.scan(
                          # lambda函数的参数顺序为sequences, prior results, non-sequences, 上述任何一个为集合则展开
                          fn=lambda coefficient, power, prior_sum, free_variable: 
                           [coefficient * (free_variable ** power), prior_sum + coefficient * (free_variable ** power)],
                          outputs_info=[None, T.zeros_like(x)], # 第一个输出非累积，无需feedback回scan；第二个输出需要accumulation
                          sequences=[coefficients, T.arange(max_coefficients_supported)], # scan取决于这里最短的sequence，自动截断
                          non_sequences=x)
final_result = [result[0].sum(), result[1][-1]]

# Compile a function
calculate_polynomial = theano.function(inputs=[coefficients, x], outputs=final_result, updates=updates)

# Test
test_coefficients = numpy.array([1, 0, 2])
test_value = 3
print(calculate_polynomial(test_coefficients, test_value))
print(1.0 * (3 ** 0) + 0.0 * (3 ** 1) + 2.0 * (3 ** 2))
