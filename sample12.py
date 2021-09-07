#12章 可変長の引数（改善編）

from step12 import Variable
from step12 import square
from step12 import exp
from step12 import Add
from step12 import add
import numpy as np

# 12.1章 可変長引数を利用して順伝搬処理を実行する
x0 = Variable(np.array(2))
x1 = Variable(np.array(3))
f = Add()
y = f(x0, x1)
print(y.data)

# 12.3章 Add関数の実装
x0 = Variable(np.array(4))
x1 = Variable(np.array(7))
y = add(x0, x1)
print(y.data)