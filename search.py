def binary_search(array, query):
    lo, hi =0,len(array)-1
    while lo<=hi:
        mid = lo+(hi-lo)//2
        val =array[mid]
        if val ==query:
            return mid
        elif val < query:
            lo = mid +1
        else:
            hi = mid -1
    return None

def firstOccurance(array, query):
    lo, hi = 0,len(array)-1
    while lo<=hi:
        mid =lo+(hi-lo)//2
        if (mid ==0 and array[mid]==query) or (array[mid]==query and array[mid-1] < query):
            return mid
        elif array[mid]<=query:
            lo=mid+1
        else:
            hi=mid-1


def lastOccurance(array,query):
    lo,hi=0,len(array)-1
    while lo<=hi:
        mid = lo + (hi-lo)//2
        if (array[mid]==query and mid ==len(array)-1) or (array[mid]==query and array[mid+1]>query):
            return mid
        elif array[mid]<=query:
            lo=mid+1
        else:
            hi=mid-1

def bsearch(t,A):
    L,U=0,len(A)-1
    while L<=U:
        M=(L+U)//2
        if A[M]<t:
            L=M+1
        elif A[M]==t:
            return M
        else:
            U=M-1
    return -1

import bisect
import collections

# @include
Student = collections.namedtuple('Student', ('name', 'grade_point_average'))
def comp_gpa(student):
    return (-student.grade_point_average, student.name)


def search_student(students, target, comp_gpa):
    i = bisect.bisect_left([comp_gpa(s) for s in students], comp_gpa(target))
    return 0 <= i < len(students) and students[i] == target
# @exclude

def search_first_of_k(A,k):
    left,right, result=0,len(A)-1,-1
    while left<=right:
        mid=(left+right)//2
        if A[mid]>k:
            right=mid-1
        elif A[mid]==k:
            result=mid
            right=mid-1
        else:
            left=mid+1
    return result

def search_entry_equal_to_its_index(A):
    left,right=0,len(A)-1
    while left<=right:
        mid=(left+right)//2
        difference=A[mid]-mid
        if difference==0:
            return mid
        elif difference>0:
            right=mid-1
        else:
            left=mid+1
    return -1

def search_smallest(A):
    left,right=0,len(A)-1
    while left<right:
        mid=(left+right)//2
        if A[mid]>A[right]:
            left=mid+1
        else:
            right=mid
    return left

def square_root(k):
    left,right=0,k
    while left<=right:
        mid=(left+right)//2
        mid_squared=mid*mid
        if mid_squared<=k:
            left=mid+1
        else:
            right=mid-1
    return left-1

import math
def square_root(x):
    left,right=(x,1.0) if x<1.0 else (1.0,x)
    while not math.isclose(left,right):
        mid=0.5*(left+right)
        mid_squared=mid*mid
        if mid_squared>x:
            right =mid
        else:
            left=mid
    return left


def matrix_search(A,x):
    row,col=0,len(A[0])-1
    while row < len(A) and col>=0:
        if A[row][col]==x:
            return True
        elif A[row][col]<x:
            row+=1
        else:
            col-=1
    return False

MinMax = collections.namedtuple('MinMax', ('smallest', 'largest'))

def find_min_max(A):
    def min_max(a,b):
        return MinMax(a,b) if a<b else MinMax(b,a)
    if len(A)<=1:
        return MinMax(A[0],A[0])
    global_min_max =min_max(A[0],A[1])
    for i in range(2,len(A)-1,2):
        local_min_max=min_max(A[i],A[i+1])
        global_min_max=MinMax(
            min(global_min_max.smallest,local_min_max.smallest),
            max(global_min_max.largest,local_min_max.largest)
        )
    if len(A)%2:
        global_min_max = MinMax(
            min(global_min_max.smallest, A[-1]),
            max(global_min_max.largest, A[-1])
        )
    return global_min_max

import heapq
import operator
import sys
import random
def find_kth_largest(k,A):
    def find_kth(comp):
        def partition_around_pivot(left, right, pivot_idx):
            pivot_value=A[pivot_idx]
            new_pivot_idx=left
            A[pivot_idx],A[right]=A[right],A[pivot_idx]
            for i in range(left,right):
                if comp(A[i],pivot_value):
                    A[i],A[new_pivot_idx]=A[new_pivot_idx],A[i]
                    new_pivot_idx+=1
                A[right],A[new_pivot_idx]=A[new_pivot_idx],A[right]
            return new_pivot_idx
        left,right=0,len(A)-1

        while left<=right:
            pivot_idx=random.randint(left,right)
            new_pivot_idx=partition_around_pivot(left,right ,pivot_idx)
            if new_pivot_idx==k-1:
                return A[new_pivot_idx]
            elif new_pivot_idx>k-1:
                right=new_pivot_idx-1
            else:
                left=new_pivot_idx+1
        raise IndexError("no kth")
    return find_kth(operator.gt)

def find_kth_smallest(k, A):
    def find_kth(comp):
        def partition_around_pivot(left, right, pivot_idx):
            pivot_value = A[pivot_idx]
            new_pivot_idx = left
            A[pivot_idx], A[right] = A[right], A[pivot_idx]
            for i in range(left, right):
                if comp(A[i], pivot_value):
                    A[i], A[new_pivot_idx] = A[new_pivot_idx], A[i]
                    new_pivot_idx += 1
            A[right], A[new_pivot_idx] = A[new_pivot_idx], A[right]
            return new_pivot_idx

        left, right = 0, len(A) - 1
        while left <= right:
            # Generates a random integer in [left, right].
            pivot_idx = random.randint(left, right)
            new_pivot_idx = partition_around_pivot(left, right, pivot_idx)
            if new_pivot_idx == k - 1:
                return A[new_pivot_idx]
            elif new_pivot_idx > k - 1:
                right = new_pivot_idx - 1
            else:  # new_pivot_idx < k - 1.
                left = new_pivot_idx + 1
        raise IndexError('no k-th node in array A')

    return find_kth(operator.lt)

def find_missing_element(ifs):
    NUM_BUCKET=1<<16
    counter=[0]*NUM_BUCKET
    for x in map(int,ifs):
        upper_part_x=x>>16
        counter[upper_part_x]+=1
    BUCKET_CAPACITY=1<<16
    candidate_bucket=next(
        i for i,c in enumerate(counter) if c < BUCKET_CAPACITY
    )
    ifs.seek(0)
    bit_vec=[0]*BUCKET_CAPACITY
    for x in map(int,ifs):
        upper_part_x=x>>16
        if candidate_bucket ==upper_part_x:
            lower_part_x=((1<<16)-1) & x
            bit_vec[lower_part_x]=1
    for i,v in enumerate(bit_vec):
        if v==0:
            return (candidate_bucket<<16) | i
    raise ValueError("no missing")

import functools
# @include
DuplicateAndMissing = collections.namedtuple('DuplicateAndMissing',
                                             ('duplicate', 'missing'))

def find_duplicate_missing(A):
    miss_XOR_dup=functools.reduce(lambda v,i: v^i[0]^i[1], enumerate(A),0)

    differ_bit,miss_or_dup=miss_XOR_dup & (~(miss_XOR_dup-1)),0
    for i,a in enumerate(A):
        if i & differ_bit:
            miss_or_dup^=i
        if a & differ_bit:
            miss_or_dup^=a
    if any(a==miss_or_dup for a in A):
        return DuplicateAndMissing(miss_or_dup,miss_or_dup^miss_XOR_dup)
    return DuplicateAndMissing(miss_or_dup^miss_XOR_dup, miss_or_dup)

