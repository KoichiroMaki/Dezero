#10章 可変長の引数（順伝搬編）

from step11 import Variable
from step11 import square
from step11 import exp
from step11 import Add
import numpy as np

# 可変長引数を利用して順伝搬処理を実行する
xs = [Variable(np.array(2)), Variable(np.array(3))]
f = Add()
ys = f(xs)
y = ys[0]
print(y.data)

# 問題）クラス利用者がめんどくさいと思ってしまう
# 原因１：Addクラスの利用者に入力変数としてリストを用意させている
# 原因２：関数の戻り値がタプル
