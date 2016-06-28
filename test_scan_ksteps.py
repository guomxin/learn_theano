# -*- encoding: utf-8 -*-

import theano
import theano.tensor as T

"""
下面的代码实现如下的逻辑：
result_A = 1
result_B = 0
for i in range(k)
    result_C = result_A + result_B
    result_A = result_A * A
    result_B = result_B + B
result = [result_A, result_B, result_C]
"""

k = T.iscalar("k")
A = T.vector("A")
B = T.vector("B")
 
result, updates = theano.scan(
                        # lambda函数的参数顺序为sequences, prior results, non-sequences, 上述任何一个为集合则展开
                        # lambda函数的返回值包含的元素个数决定result的元素个数，result的每个元素皆为List包含历次迭代的结果
                        fn=lambda prior_result_A, prior_result_B, vec_A, vec_B: [prior_result_A * vec_A, prior_result_B + vec_B, prior_result_A + prior_result_B],
                        # 输出元素，与lambda函数的输出元素个数匹配，如果某个输出不需要feedback到循环中，则指定为None；否则为迭代的初始值
                        outputs_info=[T.ones_like(A), T.zeros_like(B), None], 
                        non_sequences=[A,B], # 每次循环，值不发生变化，注意这里包含两个元素，对应上面lambda函数中的两个参数
                        n_steps=k) # 指定循环次数

# result中记录每个输出元素每次循环后的取值，最后一次循环的计算结果为最终结果
final_result = [item[-1] for item in result]
power = theano.function(inputs=[A, B, k], outputs=final_result, updates=updates)

print(power(range(10), range(10),  3))
