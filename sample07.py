from step07 import Variable
from step07 import Square
from step07 import Exp
import numpy as np

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

# 7.1 逆伝搬の自動化のために
# 逆向きに計算グラフのノードをたどる
assert y.creator == C
assert y.creator.input == b
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a
assert y.creator.input.creator.input.creator == A
assert y.creator.input.creator.input.creator.input == x

# 7.2 逆伝搬を試す
y.grad = np.array(1.0)
C = y.creator #1.関数を取得
b = C.input #2.関数の入力を取得
b.grad = C.backward(y.grad) #3.関数のbackwardメソッドを呼ぶ

B = b.creator#1.関数を取得
a = B.input#2.関数の入力を取得
a.grad = B.backward(b.grad)#3.関数のbackwardメソッドを呼ぶ

A = a.creator#1.関数を取得
x = A.input#2.関数の入力を取得
x.grad = A.backward(a.grad)#3.関数のbackwardメソッドを呼ぶ
print(x.grad)

# 7.3 backwardメソッドの追加
A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

y.grad = np.array(1.0)
y.backward()# 逆伝搬
print(x.grad)
