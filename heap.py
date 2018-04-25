from heapq import heappush, heappop, heapreplace, heapify
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

def mergeKLists(lists):
    dummy = node = ListNode(0)
    h=[(n.val, n) for n in lists if n]
    heapify(h)
    while h:
        v,n=h[0]
        if n.next is None:
            heappop(h)
        else:
            heapreplace(h,(n.next.val,n.next))
        node.next=n
        node=node.next

    return dummy.next

import heapq

def get_skyline(LRH):
    skyline, live =[], []
    i,n=0,len(LRH)
    while i<n  or live:
        if not live or i< n and LRH[i][0] <= -live[0][1]:
            x=LRH[i][0]
            while i<n and LRH[i][0] == x:
                heapq.heappush(live,(-LRH[i][2], -LRH[i][1]))
                i+=1
        else:
            x=-live[0][1]
            while live and -live[0][1] <=x:
                heapq.heappop(live)
        height = len(live) and -live[0][0]
        if not skyline or height != skyline[-1][1]:
            skyline += [x, height],
    return skyline

from collections import deque
def max_sliding_window(nums,k):
    if not nums:
        return nums

    queue = deque()
    res=[]
    for num in nums:
        if len(queue)<k:
            queue.append(num)
        else:
            res.append(max(queue))
            queue.popleft()
            queue.append(num)
    res.append(max(queue))
    return res

#######################################################################################################################
#EPI
import collections
import itertools
import sys
import random
import heapq
def top_k(k,stream):
    min_heap=[(len(s),s) for s in itertools.islice(stream,k)]
    heapq.heapify(min_heap)
    for next_string in stream:
        heapq.heappush(min_heap, (len(next_string),next_string))
    return [p[1] for p in heapq.nsmallest(k,min_heap)]

def merge_sorted_arrays(sorted_array):
    min_heap=[]
    sorted_array_iters=[iter(x) for x in sorted_array]
    for i,it in enumerate(sorted_array_iters):
        first_element =next(it,None)
        if first_element is not None :
            heapq.heappush(min_heap, (first_element,i))
    result=[]
    while min_heap:
        smallest_entry, smallest_array_i =heapq.heappop(min_heap)
        smallest_array_iter=sorted_array_iters[smallest_array_i]
        result.append(smallest_entry)
        next_element=next(smallest_array_iter,None)
        if next_element is not None:
            heapq.heappush(min_heap,(next_element,smallest_array_i))
    return result

def sort_k_increasing_decreasing_array(A):
    sorted_subarrays=[]
    INCREASING, DECREASING=range(2)
    subarray_type=INCREASING
    start_idx=0
    for i in range(1,len(A)+1):
        if (i==len(A) or
            (A[i-1]<A[i] and subarray_type==DECREASING) or
            (A[i-1]>=A[i] and subarray_type==INCREASING)):
            start_idx=i
            subarray_type=(DECREASING if subarray_type==INCREASING else INCREASING)
    return merge_sorted_arrays(sorted_subarrays)

result=[]
def sort_approximately_sorted_array(sequence, k):
    min_heap=[]
    for x in itertools.islice(sequence,k):
        heapq.heappush(min_heap,x)
    for x in sequence:
        smallest=heapq.heappushpop(min_heap,x)
        print(smallest)
        result.append(smallest)
    while min_heap:
        smallest=heapq.heappop(min_heap)
        print(smallest)
        result.append(smallest)
class Star:

    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

    @property
    def distance(self):
        return self.x**2 + self.y**2 + self.z**2

    def __lt__(self, rhs):
        return self.distance < rhs.distance
    # @exclude

    def __str__(self):
        return ' '.join(map(str, (self.x, self.y, self.z)))

    def __eq__(self, rhs):
        return math.isclose(self.distance, rhs.distance)

    # @include

def find_closet_k_stars(k,stars):
    max_heap=[]
    reader =cvs.reader(stars)
    for line in reader:
        star=Star(*map(float,line))
        heapq.heappush(max_heap,(-star.distance,star))
        if len(max_heap)==k+1:
            heapq.heappop(max_heap)
    return [s[1] for s in heapq.nlargest(k,max_heap)]

global_result=[]
def online_median(sequence):
    min_heap=[]
    max_heap=[]
    for x in sequence:
        heapq.heappush(max_heap,-heapq.heappushpop(min_heap,x))
        if len(max_heap)>len(min_heap):
            heapq.heappush(min_heap,-heapq.heappop(max_heap))
        global_result.append(0.5*(min_heap[0]+(-max_heap[0])) if len(min_heap)==len(max_heap) else min_heap[0])
        print(0.5*(min_heap[0]+(-max_heap[0])) if len(min_heap)==len(max_heap) else min_heap[0])

def k_largest_in_binary_heap(A,k):
    if k<=0:
        return []
    candidate_max_heap=[]
    candidate_max_heap.append((-A[0],0))
    result=[]
    for _ in range(k):
        candidate_idx = candidate_max_heap[0][1]
        result.append(-heapq.heappop(candidate_max_heap)[0])

        left_child_idx=2*candidate_idx+1
        if left_child_idx<len(A):
            heapq.heappush(candidate_max_heap,(-A[left_child_idx],left_child_idx))
        right_child_idx=2*candidate_idx+2
        if right_child_idx<len(A):
            heapq.heappush(candidate_max_heap,(-A[right_child_idx],right_child_idx))
    return result

class Stack:

    def __init__(self):
        self._timestamp=0
        self._max_heap=[]
    def push(self,x):
        heapq.heappush(self._max_heap,(-self._timestamp,x))
        self._timestamp+=1
    def pop(self):
        if not self._max_heap:
            raise IndexError("empty")
        return heapq.heappop(self._max_heap)[1]
    def peek(self):
        return self._max_heap[0][1]

class Queue:

    def __init__(self):
        self._timestamp=0
        self._min_heap=[]

    def enqueue(self,x):
        heapq.heappush(self._min_heap,(self._timestamp,x))
        self._timestamp+=1
    def dequeue(self):
        if not self._min_heap:
            raise IndexError("empty")
        return heapq.heappop(self._min_heap)[1]
    def head(self):
        return self._min_heap[0][1]

