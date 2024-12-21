import sys
sys.path.append("./")
import numpy as np
from bitarray import bitarray
import time

REP_NUM = 1000000
VEC_SIZE = 128

start = time.time()

total = 0
for i in range(REP_NUM):
    total += i
print(total)

print("only for roop", time.time() - start)
start = time.time()

# total_arr = np.zeros(VEC_SIZE)
# for i in range(REP_NUM):
#     total_arr += np.ones(VEC_SIZE)
# print(total_arr[0])

# print("numpy", time.time() - start)
# start = time.time()

x = bitarray(VEC_SIZE)
x[:3] = 1
for i in range(REP_NUM):
    x >>= 1

print("bitarray", time.time() - start)
start = time.time()

print(bitarray(10) == bitarray(10))


# Create a bitarray
ba = bitarray('1010011')

# Convert bitarray to numpy array
np_array = np.frombuffer(ba.tobytes(), dtype=np.uint8)

# Since numpy doesn't have a native 1-bit type, the resulting array uses 8-bit integers.
# You may need to convert these to bool for a bit-wise interpretation:
bool_array = np.unpackbits(np_array)

print(bool_array)
