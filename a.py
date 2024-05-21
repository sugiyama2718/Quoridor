from bitarray import bitarray
from bitarray.util import ba2int

arr = bitarray(128)
for x in range(10):
    for y in range(10):
        if x > 7 or y > 7:
            continue
        #if x == 0 or x == 8 or y == 0 or y == 8:
        arr[x + 11 * y] = 1
num = ba2int(arr)
print(f"{num:#X}")
