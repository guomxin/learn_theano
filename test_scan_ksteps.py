# -*- encoding: utf-8 -*-

import theano
import theano.tensor as T

"""
下面的代码实现如下的逻辑：
result_A = 1
result_B = 0
for i in range(k)
    result_A = result_A * A
    result_B = result_B + A
result = [result_A, result_B]
"""

k = T.iscalar("k")
A = T.vector("A")
B = T.vector("B")
 
result, updates = theano.scan(
                        # lambda函数的参数顺序为sequences, prior results, non-sequences, 上述任何一个为集合则展开
                        fn=lambda prior_result_A, prior_result_B, vec_A, vec_B: [prior_result_A * vec_A, prior_result_B + vec_B],
                        # 指定结果初始值，注意这里包含两个元素，对应上面lambda函数中分别有两个参数；同时表示result由两个元素组成
                        outputs_info=[T.ones_like(A), T.zeros_like(B)], 
                        non_sequences=[A,B], # 每次循环，值不发生变化，注意这里包含两个元素，对应上面lambda函数中的两个参数
                        n_steps=k) # 指定循环次数

# result中记录每个输出元素每次循环后的取值，最后一次循环的计算结果为最终结果
final_result = [item[-1] for item in result]
power = theano.function(inputs=[A, B, k], outputs=final_result, updates=updates)

print(power(range(10), range(20),  3))
