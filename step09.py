import numpy as np

class Variable:
    def __init__(self, data):
        #9.3章 ndarrayだけを扱う(Variableのデータ型は、必ずnd.array)
        if data is not None:# データ登録の状態チェック
            if not isinstance(data, np.ndarray):# 型チェック(ndarray)
                raise TypeError('{} is not surpported'.format(type(data)))# 登録データがndarray以外の場合、例外をThrow
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        #9.2章 backwardメソッドの簡略化：y.grad(np.array(1.0))の記述を省略するため
        if self.grad is None:
            self.grad = np.ones_like(self.data)# Variableのデータとgradのデータ型を同じにするため、np.ones_likeを用いて32bit浮動小数点のデータを持つようにしている

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)

# 引数がスカラであった場合、ndarrayインスタンスに変換する(=配列にして返す)
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))# 順伝搬の出力結果yをndarray型で保証する
        output.set_creator(self)
        self.input = input
        self.output = output
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

#9.1章 Squareのクラス利用を簡易化するため、関数定義する
def square(x):
    f = Square()
    return f(x)

def exp(x):
    f = Exp()
    return f(x)

