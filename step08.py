import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    #step07の「再帰を使った実装」から「ループを使った実装」に書き変える
    #要点はfuncsリストへ処理すべき関数を順に追加していること
    def backward(self):
        funcs = [self.creator]
        while funcs:
            f = funcs.pop() #関数を取得
            x, y = f.input, f.output #関数の入出力を取得
            x.grad = f.backward(y.grad) #backwardメソッドを呼ぶ

            if x.creator is not None:
                funcs.append(x.creator) #1つ前の関数をリストに追加
                
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