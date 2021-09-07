import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not surpported'.format(type(data)))
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    # 12.1 １つ目の改善：関数を使いやすく
    # (改善前)リスト=＞Addクラス=＞タプル
    # (改善後)直接複数変数=＞Addクラス=＞1変数
    def __call__(self, *inputs): # ①引数にアスタリスクをつけることで、リストを使わずに可変長引数を定義
        xs = [x.data for x in inputs]
        # forward関数呼び出し時の引数に*をつけてアンパック（※）する
        # ※）アンパック：複数の要素を持つものを分解して各変数に代入すること
        # xs=[x0, x1]の時、self.forward(*xs)とself.forward(x0, x1)は同じ
        ys = self.forward(*xs) # ③アスタリスクをつけてアンパッキング

        # forwardの戻り値がタプルでない場合、タプルに変更する
        if not isinstance(ys, tuple): # ④タプルでないデータはタプルに置換
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        # ②outputsの要素が１つの場合にはリストではなく要素のみを返す。
        return outputs if len(outputs) > 1 else outputs[0] # ３項演算子（値1 if 条件 else 値2）＝条件成立時は値1,条件未成立時は値2

        def forward(self, xs):
            raise NotImplementedError()

        def backward(self, gys):
            raise NotImplementedError()

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

class Add(Function):

    def forward(self, x0, x1):
        y = x0 + x1
        return y

def square(x):
    f = Square()
    return f(x)

def exp(x):
    f = Exp()
    return f(x)

def add(x0, x1):
    return Add()(x0, x1)
