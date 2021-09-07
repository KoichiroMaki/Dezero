#9章 現在の関数をより使いやすくするために関数に対して3つの改善をおこなう

from step09 import Variable
from step09 import square
from step09 import exp
import numpy as np

print('9.1章 クラス定義したSquare,Expを関数定義してクラス呼び出しを簡易化')
x = Variable(np.array(0.5))
y = square(exp(square(x)))
y.grad = np.array(1.0)
y.backward()
print(x.grad)

print('9.2章 backwardメソッドの簡略化')
x = Variable(np.array(0.5))
y = square(exp(square(x)))
y.backward()
print(x.grad)

print('9.3章 ndarrayだけを扱う')
x = Variable(np.array(1.0))# OK
x = Variable(None)# OK

#x = Variable(1.0)# NG:エラー

# OKケース(1次元のndarrayを二乗してもndarray型)
x = np.array([1.0])
y = x ** 2
print(type(x), x.ndim)
print(type(y))

# NGケース(0次元のndarrayを二乗するnp.float64型になる)
# ※numpy仕様でcupyでは発生しない
x = np.array(1.0)
y = x ** 2
print(type(x), x.ndim)
print(type(y))

