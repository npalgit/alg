def addWithoutOperator(x, y):
    while y != 0:
        carry = x & y
        x = x ^ y
        y = carry << 1
    return x

from collections import deque


def int_to_byes_big_endian(num):
    bytestr = deque()
    while num > 0:
        bytestr.appendleft(num & 0xff)
        num >>= 8
    return bytes(bytestr)


def int_to_bytes_little_endian(num):
    bytestr = []
    while num > 0:
        bytestr.append(num & 0xff)
        num >>= 8
    return bytes(bytestr)


def bytes_big_endian_to_int(bytestr):
    num = 0
    for b in bytestr:
        num <<= 8
        num += b
    return num


def bytes_little_endian_to_int(bytestr):
    num = 0
    e = 0
    for b in bytestr:
        num += b << e
        num += b
    return num


def count_ones(n):
    if n < 0:
        return
    counter = 0
    while n:
        counter += n & 1
        n >>= 1
    return counter


def find_missing_number(nums):
    missing = 0
    for i, num in enumerate(nums):
        missing ^= num
        missing ^= i + 1
    return missing


def is_power_of_two(n):
    return n > 0 and not n & (n - 1)


def reverse_bits(n):
    m, i = 0, 0
    while i < 32:
        m = (m << 1) + (n & 1)
        n >>= 1
        i += 1
    return m


def single_number(nums):
    i = 0
    for num in nums:
        i ^= num
    return i


def single_number2(nums):
    res = 0
    for i in range(0, 32):
        count = 0
        for num in nums:
            if (num >> i) & 1:
                count += 1
        res |= (count % 3) << i
    if res >= 2 ** 31:
        res -= 2 ** 32
    return res


def single_number3(nums):
    ones, twos = 0, 0
    for i in range(len(nums)):
        ones = (ones ^ nums[i]) & ~twos
        twos = (twos ^ nums[i]) & ~ones
    return ones


def subsets(nums):
    nums.sort()
    total = 2 ** len(nums)
    res = [] * total
    for i in range(total):
        res.append([])

    for i in range(len(nums)):
        for j in range(total):
            if ((j >> i) & 1) > 0:
                res[j].append(nums[i])
    return res


def subsets2(nums):
    res = []
    nums.sort()
    for i in range(1 << len(nums)):
        tmp = []
        for j in range(len(nums)):
            if i & 1 << j:
                tmp.append(nums[j])
        res.append(tmp)
    return res


########################################################################################################################
# EPI
import sys
import random
import math

def count_bits(x):
    num_bits = 0
    while x:
        num_bits += x & 1
        x >>= 1
    return num_bits


def parity(x):
    x ^= x >> 32
    x ^= x >> 16
    x ^= x >> 8
    x ^= x >> 4
    x ^= x >> 2
    x ^= x >> 1
    return x & 0x1


def swap_bits(x, i, j):
    if (x >> i) & 1 != (x >> j) & 1:
        bit_mask = (1 << i) | (1 << j)
        x ^= bit_mask
    return x


def closest_int_same_bit_count(x):
    NUM_UNSIGNED_BITS = 64
    for i in range(NUM_UNSIGNED_BITS - 1):
        if ((x >> i) & 1) != ((x >> (i + 1)) & 1):
            x ^= (1 << i) | (1 << (i + 1))
            return x
    raise ValueError("All bits are 0 or 1")


def multiply(x, y):
    def add(a, b):
        running_sum, carryin, k, temp_a, temp_b = 0, 0, 1, a, b
        while temp_a or temp_b:
            ak, bk = a & k, b & k
            carryout = (ak & bk) | (ak & carryin) | (bk & carryin)
            running_sum |= ak ^ bk ^ carryin
            carryin, k, temp_a, temp_b = carryout << 1, k << 1, temp_a >> 1, temp_b >> 1
        return running_sum | carryin

    running_sum = 0
    while x:
        if x & 1:
            running_sum = add(running_sum, y)
        x, y = x >> 1, y << 1
    return running_sum

def divide_bsearch(x,y):
    if x<y:
        return 0
    power_left, power_right , power_mid =0,x.bit_length(),-1
    while power_left < power_right:
        tmp=power_mid
        power_mid = power_left + (power_right-power_left)//2
        if tmp==power_mid:
            break
        yshift = y << power_mid
        if yshift > x:
            power_right = power_mid
        elif yshift <x:
            power_left=power_mid
        else:
            return 1<< power_mid
    part = 1<< power_left
    return part | divide_bsearch(x-(y<<power_left), y)

def divide(x,y):
    result, power=0,32
    y_power=y<<power
    while x>=y:
        while y_power >x:
            y_power>>=1
            power-=1

        result +=1 << power
        x-=y_power
    return result

def power_x_y(x,y):
    result, power=1.0, y
    if y<0:
        power,x=-power,1.0/x
    while power:
        if power &1:
            result*=x
        x,power=x*x, power>>1
    return result

def reverse(x):
    result, x_remaining=0,abs(x)
    while x_remaining:
        result =result *10 + x_remaining%10
        x_remaining//=10
    return -result if x<0 else result

def is_palindrome_numer(x):
    if x<0:
        return False
    num_digits =math.floor(math.log10(x))+1
    msd_mask=10**(num_digits-1)
    for i in range(num_digits//2):
        if x//msd_mask!=x%10:
            return False
        x%=msd_mask
        x//=10
        msd_mask//=100
    return True


def zero_one_random():
    return random.randrange(2)

def uniform_random(lower_bound, upper_bound):
    number_of_outcomes = upper_bound-lower_bound+1
    while True:
        result,i=0,0
        while (1<<i) < number_of_outcomes:
            result =(result<<1)| zero_one_random()
            i+=1
        if result<number_of_outcomes:
            break
    return result + lower_bound


import collections

# @include
Rectangle = collections.namedtuple('Rectangle', ('x', 'y', 'width', 'height'))

def intersect_rectangele(R1,R2):
    def is_intersect(R1,R2):
        return R1.x <=R2.x + R2.width and R1.x +R1.width>=R2.x and R1.y<=R2.y+R2.height and R1.y+R1.height>=R2.y
    if not is_intersect(R1,R2):
        return Rectangle(0,0,-1,-1)
    return Rectangle(
        max(R1.x,R2.x),
        max(R1.y,R2.y),
        min(R1.x+R1.width, R2.x+R2.width)-max(R1.x,R2.x),
        min(R1.y+R1.height, R2.y+R2.height)-max(R1.y,R2.y)
    )