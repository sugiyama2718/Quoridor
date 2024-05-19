import ctypes
import time

# 共有ライブラリをロード
lib = ctypes.CDLL('./libexample.dll')

# 関数の引数と戻り値の型を指定
add = lib.add
add.argtypes = [ctypes.c_longlong, ctypes.c_longlong]
add.restype = ctypes.c_bool

# 関数を呼び出し
result = add(10, 20)
print('The result is:', result)
time.sleep(5)
