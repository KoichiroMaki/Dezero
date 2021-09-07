import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        f = self.creator#1.関数を取得
        if f is not None:
            x = f.input#2.関数の入力を取得
            x.grad = f.backward(self.grad)#3.関数のbackwardメソッドを呼ぶ
            x.backward()#自分より１つ前の変数のbackwardメソッドを呼ぶ(再帰)


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self) # 出力変数に生みの親を覚えさせる
        self.input = input
        self.output = output # 出力も覚える
        return output

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx