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
    # 変数をリストに入れて扱うよう変更
    # (入力をリストにし複数変数を扱えるようになり、出力をタプルにしたことで複数の変数に対応する)
    def __call__(self, inputs):# 引数をリストに変更
        # inputsリストの各要素xに対してそれぞれのデータ(x.data)を取り出し、
        # その要素からなる新しいリストをxsに格納する(リスト内包表現)
        xs = [x.data for x in inputs]
        ys = self.forward(xs)
        # ysリストの各要素yに対してそれぞれのデータy(ndarray型に変換)を取り出しして、
        # その要素からなる新しいリストをoutputsに格納する(リスト内包表現)
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs# 戻り値をリストに変更

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

    def forward(self, xs):
        """
        リストの要素を加算する

        Parameters
        ----------
        xs : list of ndarray
            加算対象のリスト

        Returns
        -------
        (y,) : list of ndarray
            加算結果をリストで返却
        """
        #2つの要素を持つxsの0番目要素をx0,1番目要素をx1に代入
        x0, x1 = xs
        #xsの要素を加算する
        y = x0 + x1
        #加算結果をタプル(2つめの要素はブランク)で返す
        return (y,)

def square(x):
    f = Square()
    return f(x)

def exp(x):
    f = Exp()
    return f(x)

