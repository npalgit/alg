def josepheus(A, d=3):
    d -= 1
    idx = 0
    n = len(A)
    while n > 0:
        idx = (d + idx) % n
        print(A.pop(idx)) #A.pop(idx)
        n -= 1

def list_flattern(l, res=None):
    res = list(res) if isinstance(res, (list, tuple)) else []
    for i in l:
        if isinstance(i, (list, tuple)):
            res = list_flattern(i, res)  # recursion
        else:
            res.append(i)
    return res

def garage(initial, final):
    steps = 0
    while initial != final:
        zero = initial.index(0)
        if zero != final.index(0):
            car_to_move = final[zero]
            pos = initial.index(car_to_move)
            initial[zero], initial[pos] = initial[pos], initial[zero]
        else:
            for i in range(len(initial)):
                if initial[i] != final[i]:
                    initial[zero], initial[i] = initial[i], initial[zero]
                    break
        steps += 1
    return steps

def longest_non_repeat(string):
    if string is None:
        return 0
    temp = []
    max_len = 0
    for i in string:
        if i in temp:
            temp = []
        temp.append(i)
        max_len = max(max_len, len(temp))
    return max_len


def longest_non_repeat_two(string):
    if string is None:
        return 0
    start, max_len = 0, 0
    used_char = {}
    for index, char in enumerate(string):
        if char in used_char and start <= used_char[char]:
            start = used_char[char] + 1
        else:
            max_len = max(max_len, index - start + 1)
        used_char[char] = index

    return max_len

class Interval(object):
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = e


def merge(intervals):
    out = []
    for i in sorted(intervals, key=lambda i: i.start):
        if out and i.start <= out[-1].end:
            out[-1].end = max(out[-1].end, i.end)
        else:
            out += i,  # tuple
    return out


def print_intervals(intervals):
    res = []
    for i in intervals:
        res.append('[' + str(i.start) + ',' + str(i.end) + ']')
    print("".join(res))


def merge_intervals(l):
    if l is None:
        return None
    l.sort(key=lambda i: i[0])
    out = [l.pop(0)]
    for i in l:
        if out[-1][-1] >= i[0]:
            out[-1][-1] = max(out[-1][-1], i[-1])
        else:
            out.append(i)
    return out


given = [[1, 3], [2, 6], [8, 10], [15, 18]]
intervals = []
for l, r in given:
    intervals.append(Interval(l, r))
print_intervals(intervals)
print_intervals(merge(intervals))
print(merge_intervals(given))


def missing_ranges(nus, lo, hi):
    res = []
    start = lo
    for num in nums:
        if num < start:
            continue
        if num == start:
            start += 1
            continue
        res.append(get_range(start, num - 1))
        start = num + 1
    if start <= hi:
        res.append(get_range(start, hi))
    return res


def get_range(n1, n2):
    if n1 == n2:
        return str(n1)
    else:
        return str(n1) + "->" + str(n2)

def plusOne(digits):
    digits[-1] = digits[-1] + 1
    res = []
    ten = 0
    i = len(digits) - 1
    while i >= 0 or ten == 1:
        sum = 0
        if i >= 0:
            sum += digits[i]
        if ten:
            sum += 1
        res.append(sum % 10)
        ten = sum / 10
        i -= 1
    return res[::-1]


def plus_one(digits):
    n = len(digits)
    for i in range(n - 1, -1, -1):
        if digits[i] < 9:
            digits[i] += 1
            return digits
        digits[i] = 0
    digits.insert(0, 1)
    return digits


def plus_1(num_arr):
    for idx, digit in reversed(list(enumerate(num_arr))):
        num_arr[idx] = (num_arr[idx] + 1) % 10
        if num_arr[idx]:
            return num_arr
    return [1] + num_arr


def rotate_one_by_one(nums, k):
    n = len(nums)
    for i in range(k):
        temp = nums[n - 1]
        for j in range(n - 1, 0, -1):
            nums[j] = nums[j - 1]
        nums[0] = temp


def rotate(nums, k):
    n = len(nums)
    k = k % n
    reverse(nums, 0, n - k - 1)
    reverse(nums, n - k, n - 1)
    reverse(nums, 0, n - 1)


def reverse(array, a, b):
    while a < b:
        array[a], array[b] = array[b], array[a]
        a += 1
        b -= 1


def rotate_array(array, k):
    if array is None:
        return None
    length = len(array)
    return array[length - k:] + array[:length - k]


def summary_ranges(nums):
    res = []
    if len(nums) == 1:
        return [str(nums[0])]
    i = 0
    while i < len(nums):
        num = nums[i]
        while i + 1 < len(nums) and nums[i + 1] - nums[i] == 1:
            i += 1
        if nums[i] != num:
            res.append(str(num) + "->" + str(nums[i]))
        else:
            res.append(str(num))
        i += 1
    return res


def three_sum(nums):
    res = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        l, r = i + 1, len(nums) - 1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s > 0:
                r -= 1
            elif s < 0:
                l += 1
            else:
                res.append((nums[i], nums[l], nums[r]))
                while l < r and nums[l] == nums[l + 1]:
                    l += 1
                while l < r and nums[r] == nums[r - 1]:
                    r -= 1
                l += 1
                r -= 1

    return res


x = [-1, 0, 1, 2, -1, -4]
print(three_sum(x))


def two_sum(nums, target):
    dict = {}
    for i, num in enumerate(nums):
        if num in dict:
            return [dict[num], i]
        else:
            dict[target - num] = i


arr = [3, 2, 4]
target = 6
res = two_sum(arr, target)
print(res)

######################################################################################
# EPI
import collections
import sys
import random


def even_odd(A):
    next_even, next_odd = 0, len(A) - 1
    while next_even < next_odd:
        if A[next_even] % 2 == 0:
            next_even += 1
        else:
            A[next_even], A[next_odd] = A[next_odd], A[next_even]
            next_odd -= 1


RED, WHITE, BLUE = range(3)


def dutch_flag_partition(pivot_index, A):
    pivot = A[pivot_index]
    smaller, equal, larger = 0, 0, len(A)
    while equal < larger:
        if A[equal] < pivot:
            A[smaller], A[equal] = A[equal], A[smaller]
            smaller, equal = smaller + 1, equal + 1
        elif A[equal] == pivot:
            equal += 1
        else:
            larger -= 1
            A[equal], A[larger] = A[larger], A[equal]


def plus_one(A):
    A[-1] += 1
    for i in reversed(range(1, len(A))):
        if A[i] != 10:
            break
        A[i] = 0
        A[i - 1] += 1
    if A[0] == 10:
        A[0] = 0
        A.insert(0, 1)
    return A


def multiply(num1, num2):
    sign = -1 if (num1[0] < 0) ^ (num2[0] < 0) else 1
    num1[0], num2[0] = abs(num1[0]), abs(num2[0])
    result = [0] * (len(num1) + len(num2))
    for i in reversed(range(len(num1))):
        for j in reversed(range(len(num2))):
            result[i + j + 1] += num1[i] * num2[j]
            result[i + j] += result[i + j + 1] // 10
            result[i + j + 1] %= 10
    result = result[next((i for i, x in enumerate(result) if x != 0), len(result)):] or [0]


def can_reach_end(A):
    furthest_reach_so_far, last_index = 0, len(A) - 1
    i = 0
    while i <= furthest_reach_so_far and furthest_reach_so_far < last_index:
        furthest_reach_so_far = max(furthest_reach_so_far, A[i] + i)
        i += 1
    return furthest_reach_so_far >= last_index


def delete_duplicates(A):
    if not A:
        return 0
    write_index = 1
    for i in range(1, len(A)):
        if A[write_index - 1] != A[i]:
            A[write_index] = A[i]
            write_index += 1
    return write_index


def buy_and_sell_stock_once(prices):
    min_price_so_far, max_profit = float('inf'), 0.0
    for price in prices:
        max_profit_sell_today = price - min_price_so_far
        max_profit = max(max_profit, max_profit_sell_today)
        min_price_so_far = min(min_price_so_far, price)
    return max_profit


def buy_and_sell_stock_twice_constant_space(prices):
    min_prices, max_profits = [float('inf')] * 2, [0] * 2
    for price in prices:
        for i in reversed(list(range(2))):
            max_profits[i] = max(max_profits[i], price - min_prices[i])
            min_prices[i] = min(min_prices[i], price - (0 if i == 0 else max_profits[i - 1]))
    return max_profits[-1]


def rearrange(A):
    for i in range(len(A)):
        A[i:i + 2] = sorted(A[i:i + 2], reverse=i % 2)


def generate_primes_from_1_to_n(n):
    size = (n - 3) // 2 + 1
    primes = [2]
    is_prime = [True] * size
    for i in range(size):
        if is_prime[i]:
            p = i * 2 + 3
            primes.append(p)
            for j in range(2 * i ** 2 + 6 * i + 3, size, p):
                is_prime[j] = False
    return primes


def apply_permutation(perm, A):
    def cyclic_permutation(start, perm, A):
        i, temp = start, A[start]
        while True:
            next_i = perm[i]
            next_temp = A[next_i]
            A[next_i] = temp
            i, temp = next_i, next_temp
            if i == start:
                break

    for i in range(len(A)):
        j = perm[i]
        while j != i:
            if j < i:
                break
            j = perm[j]
        else:
            cyclic_permutation(i, perm, A)


import itertools


def next_permutation(perm):
    inversion_point = len(perm) - 2
    while inversion_point >= 0 and perm[inversion_point] >= perm[inversion_point + 1]:
        inversion_point -= 1
    if inversion_point == -1:
        return []
    for i in reversed(range(inversion_point + 1, len(perm))):
        if perm[i] > perm[inversion_point]:
            perm[inversion_point], perm[i] = perm[i], perm[inversion_point]
            break
    perm[inversion_point + 1:] = reversed(perm[inversion_point + 1:])
    return perm


def random_sampling(k, A):
    for i in range(k):
        r = random.randint(i, len(A) - 1)
        A[i], A[r] = A[r], A[i]


def online_random_sample(it, k):
    sampling_results = list(itertools.islice(it, k))
    num_seen_so_far = k
    for x in it:
        num_seen_so_far += 1
        idx_to_replace = random.randrange(num_seen_so_far)
        if idx_to_replace < k:
            sampling_results[idx_to_replace] = x
    return sampling_results


def compute_randome_permutation(n):
    permutation = list(range(n))
    random_sampling(n, permutation)
    return permutation


def online_sampling(n, k):
    changed_elements = {}
    for i in range(k):
        rand_idx = random.randrange(i, n)
        rand_idx_mapped = changed_elements.get(rand_idx, rand_idx)
        i_mapped = changed_elements.get(i, i)
        changed_elements[rand_idx] = i_mapped
        changed_elements[i] = rand_idx_mapped
    return [changed_elements[i] for i in range(k)]


import math
import bisect


def nonuniform_random_number_generation(values, probabilities):
    prefix_sum_of_probabilities = ([0.0] + list(itertools.accumulate(probabilities)))
    interval_idx = bisect.bisect(prefix_sum_of_probabilities, random.random()) - 1
    return values[interval_idx]


def is_valid_suduku(partial_assignment):
    def has_duplicate(block):
        block = list(filter(lambda x: x != 0, block))
        return len(block) != len(set(block))

    n = len(partial_assignment)
    if any(
            has_duplicate([partial_assignment[i][j]] for j in range(n)) or
            has_duplicate([partial_assignment[j][i]] for j in range(n))
            for i in range(n)
    ):
        return False

    region_size = int(math.sqrt(n))
    return all(not has_duplicate(
        [
            partial_assignment[a][b]
            for a in range(region_size * I, region_size * (I + 1))
            for b in range(region_size * J, region_size * (J + 1))
        ]
    ) for I in range(region_size) for J in range(region_size))

def matrix_in_spiral_order(square_matrix):
    SHIFT = ((0, 1), (1, 0), (0, -1), (-1, 0))
    direction = x = y = 0
    spiral_ordering = []
    for _ in range(len(square_matrix)**2):
        spiral_ordering.append(square_matrix[x][j])
        square_matrix[x][y]=0
        next_x, next_y = x+SHIFT[direction][0], y+SHIFT[direction][1]
        if next_x not in range(len(square_matrix)) or next_y not in range(len(square_matrix)) or square_matrix[next_x][next_y]==0:
            direction=(direction+1) & 3
            next_x, next_y =x+SHIFT[direction][0],y+SHIFT[direction][1]
    return spiral_ordering

import copy
def rotate_matrix(A):
    for i in range(len(A)//2):
        for j in range(i,len(A)-i-1):
            temp=A[i][j]
            A[i][j]=A[-1-j][i]
            A[-1-j][i]=A[-1-i][-1-j]
            A[-1-i][-1-j]=A[j][-1-i]
            A[j][-1-i]=temp

class RotatedMatrix:
    def __init__(self,sqaure_matrix):
        self._square_matrix=sqaure_matrix
    def read_entry(self,i,j):
        return self._square_matrix[~j][i]
    def write_entry(self,i,j,v):
        self._square_matrix[~j][i]=v

def generate_pascal_triangle(n):
    result=[[1]*(i+1) for i in range(n)]
    for i in range(n):
        for j in range(1,i):
            result[i][j]=result[i-1][j-1]+result[i-1][j]
        return result

