#10章 ソフトウェアテストとその自動化

from step10 import Variable
from step10 import square
from step10 import exp
import numpy as np
import unittest 

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


# Python標準ライブラリのunittestを用いてsuare関数のテストプログラムを実装する
class SquareTest(unittest.TestCase):# unittest.TestCaseを継承するSquareTestクラスを実装
    def test_forward(self):# 名前が「test」で始まる任意のメソッド内にテスト項目を記入
        print('10.1章 square関数の順伝搬のテスト')
        x = Variable(np.array(2.0))# ndarray型データ2.0の変数を定義
        y = square(x)# square関数へ渡し、結果をyに格納する
        expected = np.array(4.0)# 想定出力(4.0)をexpected変数に格納する
        self.assertEqual(y.data, expected)# squareの出力と期待出力が一致することを確認

    def test_backward(self):
        print('10.2章 square関数の逆伝搬のテスト')
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)
    
    def test_gradient_check(self):
        print('10.3章 勾配確認による自動テスト')
        # 逆伝搬の出力を手計算しなくとも求められるテスト
        x = Variable(np.random.rand(1))# テスト用のランダムな入力値を生成
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)# 数値微分を求める
        flg = np.allclose(x.grad, num_grad, rtol=1e-05, atol=1e-08)# 逆伝搬の出力と数値微分の出力を比較（※）
        #※）np.allclose(a,b,rtol,atol)は、aとbのすべての要素が条件(|a-b|<=(atol+rtol * |b|)を満たすかチェック（=値が近いことを確認）
        self.assertTrue(flg)




# unittestを実行するために以下のコマンドを実行
# python3 -m unittest step10.py
# なお以下のunittestのメイン関数呼び出しを記入することで、
# 本Pythonプログラムを実行するのみでテスト実行可能
unittest.main()