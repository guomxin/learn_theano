# -*- encoding: utf-8 -*-

import theano
import theano.tensor as T
import theano.ifelse as ifelse
import numpy

a_val = numpy.array([1, 0, 1, 0])
b_val = numpy.array([0, 1, 0, 1])
x_val = numpy.array([1, 1, 1, 1])
y_val = numpy.array([2, 2, 2, 2])


a,b = T.lvectors('a', 'b')
x,y = T.lvectors('x', 'y')

s1 = T.switch(T.gt(a, b), x, y)
f_s1 = theano.function([a, b, x, y], s1)
result_s1 = f_s1(a_val, b_val, x_val, y_val)
print result_s1 #结果为[1, 2, 1, 2]，进行了element-wise操作

s2 = T.switch(a_val > b_val, [2, 2, 2, 2], [3, 3, 3, 3])
f_s2 = theano.function([], s2)
result_s2 = f_s2()
print result_s2

c,d = T.lscalars('c', 'd')
if1 = ifelse.ifelse(T.gt(c, d), x, y) #相当于T.gt(c, d) ? x : y，不支持element-wise操作
                                      #第一个参数必须返回一个标量
f_if1 = theano.function([c, d, x, y], if1)
result_if1 = f_if1(2, 1, x_val, y_val)
print result_if1

if2 = ifelse.ifelse(1, x_val, y_val) # 参数为常量时
f_if2 = theano.function([], if2)
result_if2 = f_if2()
print result_if2
