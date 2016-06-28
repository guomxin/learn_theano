# -*- encoding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T

up_to = T.iscalar("up_to")

# 命名函数，代替scan中的lambda函数
def accumulate_by_adding(arange_val, sum_to_date):
    return sum_to_date + arange_val
seq = T.arange(up_to)

# 如果这里使用T.as_tensor_variable(0)，outputs_info为int8类型
# 这样计算就需要downcast, theano会报错
# T.as_tensor_variable(x)，如果x为tensor var则pass through，否则包装为tensor const类型
#outputs_info = T.as_tensor_variable(0)
outputs_info = T.as_tensor_variable(np.asarray(0, seq.dtype))
scan_result, scan_updates = theano.scan(fn=accumulate_by_adding,
                                        outputs_info=outputs_info,
                                        sequences=seq)
triangular_sequence = theano.function(inputs=[up_to], outputs=scan_result)

# test
some_num = 15
print(triangular_sequence(some_num))
print([n * (n + 1) // 2 for n in range(some_num)])
