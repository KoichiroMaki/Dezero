from step08 import Variable
from step08 import Square
from step08 import Exp
import numpy as np

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

# 8.2 ループを使った実装
y.grad = np.array(1.0)
y.backward()# 逆伝搬
print(x.grad)
