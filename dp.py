def max_profit_optimized(prices):
    cur_max=max_so_far=0
    for i in range(1,len(prices)):
        cur_max=max(0,cur_max+prices[i]-prices[i-1]) #local max
        max_so_far=max(max_so_far,cur_max) #global max
    return max_so_far

def climb_stairs_optimized(n):
    a=b=1
    for _ in range(n):
        a,b=b,a+b
    return a

def combination_sum_bottom_up(nums,target):
    comb=[0]* (target+1)
    comb[0]=1
    for i in range(0,len(comb)):
        for j in range(len(nums)):
            if i-nums[j]>=0:
                comb[i]+=comb[i-nums[j]]
    return comb[target]


def house_robber(houses):
    last,now=0,0
    for house in houses:
        tmp=now
        now=max(last+house,now)
        last=tmp

    return now


class Job:
    def __init__(self,start,finish,profit):
        self.start=start
        self.finish=finish
        self.profit =profit
#bsearch
def bsearch(job,start_index):
    lo,ho=0,start_index-1
    while lo<=hi:
        mid=(lo+hi)//2
        if job[mid].finish <=job[start_index].start:
            if job[mid+1].finish<=job[start_index].start:
                lo=mid+1
            else:
                return mid
        else:
            hi=mid-1
    return -1

def schedule(job):
    job=sorted(job,key=lambda j: j.finish) #sort
    n=len(job)
    table=[0 for _ in range(n)]
    table[0]=job[0].profit

    for i in range(1,n):
        inclProf =job[i].profit
        l=bsearch(job,i) #bsearch
        if(l!=-1):
            inclProf+=table[l]

        table[i]=max(inclProf,table[i-1])

    return table[n-1]

class Item(object):
    def __init__(self,value,weight):
        self.value=value
        self.weight=weight

def get_maximum_value(items,capacity):
    dp=[0]*(capacity+1)
    for item in items:
        dp_tmp=[total_value for total_value in dp]
        for current_weight in range(capacity+1):
            total_weight = current_weight+ item.weight
            if total_weight <=capacity:
                dp_tmp[total_weight]=max(dp_tmp[total_weight], dp[current_weight]+item.value)
        dp=dp_tmp
    return max(dp)

def longest_increasing_subsequence(sequence):
    length=len(sequence)
    counts=[1 for _ in range(length)]
    for i in range(1,length):
        for j in range(0,i):
            if sequence[i]>sequence[j]:
                counts[i]=max(counts[i],counts[j]+1)
                print(counts)
    return max(counts)

def max_product(nums):
    lmin=lmax=gmax=nums[0]
    for i in range(len(nums)):
        t1=nums[i]*lmax
        t2=nums[i]*lmin
        lmax=max(max(t1,t2),nums[i])
        lmin=min(min(t1,t2),nums[i])
        gmax=max(gmax,lmax)

from functools import reduce
def subarray_with_max_product(arr):
    l=len(arr)
    product_so_far = max_product_end =1
    max_start_i =0
    so_far_start_i = so_far_end_i =0
    all_negative_flag = True

    for i in range(l):
        max_product_end *=arr[i]
        if arr[i]>0: all_negative_flag=False
        if max_product_end<=0:
            max_product_end=arr[i]
            max_start_i=i
        if product_so_far <=max_product_end:
            product_so_far=max_product_end
            so_far_end_i=i
            so_far_start_i=max_start_i

    if all_negative_flag:
        print("max_product_so_far: %s, %s" % (reduce(lambda x, y: x * y, arr), arr))
    else:
        print("max_product_so_far: %s, %s" % (product_so_far,arr[so_far_start_i:so_far_end_i + 1]))


def max_subarray(array):
    max_so_far =max_now=array[0]
    for i in range(1,len(array)):
        max_now=max(array[i], max_now+array[i])
        max_so_far=max(max_so_far,max_now)
    return max_so_far


def num_decodings(s):
    if not s or s[0]=='0':
        return 0
    wo_last,wo_last_two =1,1
    for i in range(1,len(s)):
        x=wo_last if s[i]!="0" else 0
        y=wo_last_two if int(s[i-1:i+1])<27 and s[i-1]!='0' else 0
        wo_last_two=wo_last
        wo_last=x+y
    return wo_last

def num_decodings2(s):
    if not s or s.startswith('0'):
        return 0
    stack=[1,1]
    for i in range(1,len(s)):
        if s[i]=='0':
            if s[i-1]=='0' or s[i-1]>'2':
                return 0
            stack.append(stack[-2])
        elif 9<int(s[i-1:i+1])<27:
            stack.append(stack[-2]+stack[-1])
        else:
            stack.append(stack[-1])
    return stack[-1]


class Solution(object):
    def isMatch(self,s,p):
        m,n=len(s)+1, len(p)+1
        matches=[[False]* n for _ in range(m)]
        matches[0][0] = True
        for i,element in enumerate(p[1:],2):
            matches[0][i]=matches[0][i-2] and element=='*'
        for i,ss in enumerate(s,1):
            for j,pp in enumerate(p,1):
                if pp!='*':
                    matches[i][j]=matches[i-1][j-1] and (ss==pp or pp=='.')
                else:
                    matches[i][j]!=matches[i][j-2]

                    if ss==p[j-2] or p[j-2]=='.':
                        matches[i][j]|=matches[i-1][j]
        return matches[-1][-1]

INT_MIN=-32767
def cutRod(price,n):
    val =[0 for x in range(n+1)]
    val[0]=0
    for i in range(1,n+1):
        max_val =INT_MIN
        for j in range(i):
            max_val = max(max_val, price[j]+val[i-j-1])
        val[i]=max_val
    return val[n]


def word_break(s,word_dict):
    dp=[False]* (len(s)+1)
    dp[0]=True
    for i in range(1,len(s)+1):
        for j in range(0,i):
            if dp[j] and s[j:i] in word_dict:
                dp[i]=True
                break
    return dp[-1]

###############################################################################################
import sys
import random


# @include
def fibonacci(n):
    if n <= 1:
        return n

    f_minus_2, f_minus_1 = 0, 1
    for _ in range(1, n):
        f = f_minus_2 + f_minus_1
        f_minus_2, f_minus_1 = f_minus_1, f
    return f_minus_1

# @include
def find_maximum_subarray(A):
    min_sum = max_sum = 0
    for running_sum in itertools.accumulate(A):
        min_sum = min(min_sum, running_sum)
        max_sum = max(max_sum, running_sum - min_sum) # max -min
    return max_sum
# @exclude

def num_combinations_for_final_score(final_score, individual_play_scores):
    # One way to reach 0.
    num_combinations_for_score = [[1] + [0] * final_score
                                  for _ in individual_play_scores]
    for i in range(len(individual_play_scores)):
        for j in range(1, final_score + 1):
            without_this_play = (num_combinations_for_score[i - 1][j]
                                 if i >= 1 else 0)
            with_this_play = (
                num_combinations_for_score[i][j - individual_play_scores[i]]
                if j >= individual_play_scores[i] else 0)
            num_combinations_for_score[i][j] = (
                without_this_play + with_this_play)
    return num_combinations_for_score[-1][-1]

def levenshtein_distance(A, B):
    def compute_distance_between_prefixes(A_idx, B_idx):
        if A_idx < 0:
            # A is empty so add all of B's characters.
            return B_idx + 1
        elif B_idx < 0:
            # B is empty so delete all of A's characters.
            return A_idx + 1
        if distance_between_prefixes[A_idx][B_idx] == -1:
            if A[A_idx] == B[B_idx]:
                distance_between_prefixes[A_idx][B_idx] = (
                    compute_distance_between_prefixes(A_idx - 1, B_idx - 1))
            else:
                substitute_last = compute_distance_between_prefixes(A_idx - 1,
                                                                    B_idx - 1)
                add_last = compute_distance_between_prefixes(A_idx - 1, B_idx)
                delete_last = compute_distance_between_prefixes(A_idx,
                                                                B_idx - 1)
                distance_between_prefixes[A_idx][B_idx] = (
                    1 + min(substitute_last, add_last, delete_last))
        return distance_between_prefixes[A_idx][B_idx]

    distance_between_prefixes = [[-1] * len(B) for _ in A]
    return compute_distance_between_prefixes(len(A) - 1, len(B) - 1)



def compute_number_of_ways_space_efficient(n, m):
    if n < m:
        n, m = m, n

    A = [1] * m
    for i in range(1, n):
        prev_res = 0
        for j in range(m):
            A[j] += prev_res
            prev_res = A[j]
    return A[m - 1]


def is_pattern_contained_in_grid(grid, S):
    def is_pattern_suffix_contained_starting_at_xy(x, y, offset):
        if len(S) == offset:
            # Nothing left to complete.
            return True

        # Check if (x, y) lies outside the grid.
        if (0 <= x < len(grid) and 0 <= y < len(grid[x]) and
                grid[x][y] == S[offset] and
            (x, y, offset) not in previous_attempts and any(
                is_pattern_suffix_contained_starting_at_xy(x + a, y + b,
                                                           offset + 1)
                for a, b in ((-1, 0), (1, 0), (0, -1), (0, 1)))):
            return True
        previous_attempts.add((x, y, offset))
        return False

    # Each entry in previous_attempts is a point in the grid and suffix of
    # pattern (identified by its offset). Presence in previous_attempts
    # indicates the suffix is not contained in the grid starting from that
    # point.
    previous_attempts = set()
    return any(
        is_pattern_suffix_contained_starting_at_xy(i, j, 0)
        for i in range(len(grid)) for j in range(len(grid[i])))

import collections
# @include
Item = collections.namedtuple('Item', ('weight', 'value'))


def optimum_subjec_to_capacity(items, capacity):
    # Returns the optimum value when we choose from items[:k + 1] and have a
    # capacity of available_capacity.
    def optimum_subject_to_item_and_capacity(k, available_capacity):
        if k < 0:
            # No items can be chosen.
            return 0

        if V[k][available_capacity] == -1:
            without_curr_item = optimum_subject_to_item_and_capacity(
                k - 1, available_capacity)
            with_curr_item = (0 if available_capacity < items[k].weight else (
                items[k].value + optimum_subject_to_item_and_capacity(
                    k - 1, available_capacity - items[k].weight)))
            V[k][available_capacity] = max(without_curr_item, with_curr_item)
        return V[k][available_capacity]

    # V[i][j] holds the optimum value when we choose from items[:i + 1] and have
    # a capacity of j.
    V = [[-1] * (capacity + 1) for _ in items]
    return optimum_subject_to_item_and_capacity(len(items) - 1, capacity)
# @exclude



# @include
def decompose_into_dictionary_words(domain, dictionary):
    # When the algorithm finishes, last_length[i] != -1 indicates domain[:i +
    # 1] has a valid decomposition, and the length of the last string in the
    # decomposition is last_length[i].
    last_length = [-1] * len(domain)
    for i in range(len(domain)):
        # If domain[:i + 1] is a dictionary word, set last_length[i] to the
        # length of that word.
        if domain[:i + 1] in dictionary:
            last_length[i] = i + 1

        # If last_length[i] = -1 look for j < i such that domain[: j + 1] has a
        # valid decomposition and domain[j + 1:i + 1] is a dictionary word. If
        # so, record the length of that word in last_length[i].
        if last_length[i] == -1:
            for j in range(i):
                if last_length[j] != -1 and domain[j + 1:i + 1] in dictionary:
                    last_length[i] = i - j
                    break

    decompositions = []
    if last_length[-1] != -1:
        # domain can be assembled by dictionary words.
        idx = len(domain) - 1
        while idx >= 0:
            decompositions.append(domain[idx + 1 - last_length[idx]:idx + 1])
            idx -= last_length[idx]
        decompositions = decompositions[::-1]
    return decompositions
# @exclude


def minimum_path_weight(triangle):
    min_weight_to_curr_row = [0]
    for row in triangle:
        min_weight_to_curr_row = [
            row[j] +
            min(min_weight_to_curr_row[max(j - 1, 0)],
                min_weight_to_curr_row[min(j, len(min_weight_to_curr_row) - 1)])
            for j in range(len(row))
        ]
    return min(min_weight_to_curr_row)


# @include
def maximum_revenue(coins):
    def compute_maximum_revenue_for_range(a, b):
        if a > b:
            # No coins left.
            return 0

        if maximum_revenue_for_range[a][b] == 0:
            max_revenue_a = coins[a] + min(
                compute_maximum_revenue_for_range(a + 2, b),
                compute_maximum_revenue_for_range(a + 1, b - 1))
            max_revenue_b = coins[b] + min(
                compute_maximum_revenue_for_range(a + 1, b - 1),
                compute_maximum_revenue_for_range(a, b - 2))
            maximum_revenue_for_range[a][b] = max(max_revenue_a, max_revenue_b)
        return maximum_revenue_for_range[a][b]

    maximum_revenue_for_range = [[0] * len(coins) for _ in coins]
    return compute_maximum_revenue_for_range(0, len(coins) - 1)
# @exclude

import itertools
def maximum_revenue_alternative(coins):
    def maximum_revenue_alternative_helper(a, b):
        if a > b:
            return 0
        elif a == b:
            return coins[a]

        if maximum_revenue_for_range[a][b] == -1:
            maximum_revenue_for_range[a][b] = max(
                coins[a] + prefix_sum[b] -
                (prefix_sum[a]
                 if a + 1 > 0 else 0) - maximum_revenue_alternative_helper(
                     a + 1, b), coins[b] + prefix_sum[b - 1] -
                (prefix_sum[a - 1] if a > 0 else 0
                 ) - maximum_revenue_alternative_helper(a, b - 1))
        return maximum_revenue_for_range[a][b]

    prefix_sum = list(itertools.accumulate(coins))
    maximum_revenue_for_range = [[-1] * len(coins) for _ in coins]
    return maximum_revenue_alternative_helper(0, len(coins) - 1)


def number_of_ways_to_top(top, maximum_step):
    def compute_number_of_ways_to_h(h):
        if h <= 1:
            return 1

        if number_of_ways_to_h[h] == 0:
            number_of_ways_to_h[h] = sum(
                compute_number_of_ways_to_h(h - i)
                for i in range(1, min(maximum_step, h) + 1))
        return number_of_ways_to_h[h]

    number_of_ways_to_h = [0] * (top + 1)
    return compute_number_of_ways_to_h(top)


def minimum_messiness(words, line_length):
    num_remaining_blanks = line_length - len(words[0])
    # min_messiness[i] is the minimum messiness when placing words[0:i + 1].
    min_messiness = ([num_remaining_blanks**2] + [float('inf')] *
                     (len(words) - 1))
    for i in range(1, len(words)):
        num_remaining_blanks = line_length - len(words[i])
        min_messiness[i] = min_messiness[i - 1] + num_remaining_blanks**2
        # Try adding words[i - 1], words[i - 2], ...
        for j in reversed(range(i)):
            num_remaining_blanks -= len(words[j]) + 1
            if num_remaining_blanks < 0:
                # Not enough space to add more words.
                break
            first_j_messiness = 0 if j - 1 < 0 else min_messiness[j - 1]
            current_line_messiness = num_remaining_blanks**2
            min_messiness[i] = min(min_messiness[i],
                                   first_j_messiness + current_line_messiness)
    return min_messiness[-1]

# @include
def longest_nondecreasing_subsequence_length(A):
    # max_length[i] holds the length of the longest nondecreasing subsequence
    # of A[:i + 1].
    max_length = [1] * len(A)
    for i in range(1, len(A)):
        max_length[i] = max(1 + max(max_length[j] for j in range(i)
                                    if A[i] >= A[j]), max_length[i])
    return max(max_length)
# @exclude
