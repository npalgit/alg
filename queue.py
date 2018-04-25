from __future__ import division
from collections import deque
def max_sliding_window(nums,k):
    d = deque()
    out =[]
    for i,n in enumerate(nums):
        while d and nums[d[-1]]<n:
            d.pop()
        d.append(i)
        if d[0]==i-k:
            d.popleft() #deq: popleft or pop(0)
        if i>=k-1:
            out+=nums[d[0]],
    return out


from collections import deque
class MovingAverage(object):
    def __init__(self,size):
        self.queue=deque(maxlen=size)
    def next(self,val):
        self.queue.append(val)
        return sum(self.queue)/ len(self.queue)

class AbstractQueue:
    def __int__(self):
        self.top=0
    def isEmpty(self):
        return self.top==0
    def __len__(self):
        return self.top
    def __str__(self):
        result ='----------\n'
        for element in self:
            result+=str(element)+'\n'
        return result[:-1]+'\n-----------'

class ArrayQueue(AbstractQueue):
    def __int__(self,size=10):
        AbstractQueue.__init__(self)
        self.array=[None]*size
        self.front =0
        self.rear=0
    def enqueue(self,value):
        if self.rear == len(self.array):
            self.expand()
        self.array[self.rear]=value
        self.rear+=1
        self.top+=1
    def dequeue(self):
        if self.isEmpty():
            raise  IndexError("empty")
        value = self.array[self.front]
        self.array[self.front]=None
        self.front-=1
        self.top-=1

        return value

    def expand(self):
        new_array=[None]*len(self.array)*2
        for i,element in enumerate(self.array):
            new_array[i]=element
        self.array=new_array

    def __iter__(self):
        probe =self.rear
        while True:
            if probe <0:
                raise StopIteration
            yield self.array[probe]
            probe-=1
class QueueNode(object):
    def __init__(self,value):
        self.value =value
        self.next = None

    def enqueue(self,value):
        node=QueueNode(value)
        if not self.front:
            self.front =node
            self.rear =node
        else:
            self.rear.next =node
            self.rear =node
        self.top+=1

    def dequeue(self):
        if self.isEmpty():
            raise IndexError("empty")
        value =self.front.value
        if self.front is self.rear:
            self.front=None
            self.rear=None
        else:
            self.front=self.front.next
        self.top-=1
        return value
    def __iter__(self):
        probe=self.rear
        while True:
            if probe is None:
                raise  StopIteration
            yield probe.value
            probe = probe.next

class HeapPriorityQueue(AbstractQueue):
    def __int__(self):
        pass

def recosntruct_queue(people):
    queue=[]
    people.sort(key=lambda  x:(-x[0],x[1]))
    for h,k in people:
        queue.insert(k,(h,k))
    return queue


class ZigZagIterator:
    def __init__(self,v1,v2):
        self.queue=[_ for _ in (v1,v2) if _ ]
        print(self.queue)
    def next(self):
        v=self.queue.pop(0)
        ret=v.pop(0)
        if v: self.queue.append(v)
        return ret
    def has_next(self):
        if self.queue: return True
        return False

import collections
import sys
import random
def examine_buildings_with_sunset(it):
    BuildingWithHeight = collections.namedtuple('BuildingWithHeight',('id','height'))
    candidates=[]
    for building_idx, building_height in enumerate(it):
        while candidates and building_height>=candidates[-1].height:
            candidates.pop()
        candidates.append(BuildingWithHeight(building_idx,building_height))
    return [candidate.id for candidate in reversed(candidates)]

class Queue:
    def __init__(self):
        self._data=[]
    def enqueue(self,x):
        self._data.append(x)
    def dequeue(self):
        return self._data.pop()
    def max(self):
        return max(self._data)

def binary_tree_depth_order(tree):
    result,curr_depth_nodes=[],collections.deque([tree])
    while curr_depth_nodes:
        next_depth_nodes, this_level=collections.deque([]),[]
        while curr_depth_nodes:
            curr=curr_depth_nodes.popleft()
            if curr:
                this_level.append(curr.data)
                next_depth_nodes+=[curr.left,curr.right]
        if this_level:
            result.append(this_level)
        curr_depth_nodes=next_depth_nodes
    return result

class Queue:
    SCALE_FACTOR=2
    def __init__(self,capacity):
        self._entries=[None]*capacity
        self._head=self._tail=self._num_queue_elements=0
    def enqueue(self,x):
        if self._num_queue_elements==len(self._entries):
            self._entries=(self._entries[self._head:]+self._entries[:self._head])
            self._head,self_tail=0,self._num_queue_elements
            self._entries+=[None]*(len(self._entries)*Queue.SCALE_FACTOR)
        self._entries[self._tail]=x
        self._tail=(self._tail+1)%len(self._entries)
        self._num_queue_elements+=1
    def dequeue(self):
        if not self._num_queue_elements:
            raise IndexError("empty")
        self._num_queue_elements-=1
        ret=self._entries[self._head]
        self._head=(self._head+1)%len(self._entries)
        return ret

    def size(self):
        return self._num_queue_elements

class Queue:
    def __init__(self):
        self._enq, self._deq=[],[]
    def enqueue(self,x):
        self._enq.append(x)
    def dequeue(self):
        if not self._deq:
            while self._enq:
                self._deq.append(self._enq.pop())
        if not self._deq:
            raise IndexError("empty")
        return self._deq.pop()

from stack import Stack
class QueueWithMax:
    def __init__(self):
        self._enquue=Stack()
        self._dequeue=Stack()
    def enqueue(self,x):
        self._enquue.push(x)
    def dequeue(self):
        if self._dequeue.empty():
            while not self._enquue.empty():
                self._dequeue.push(self._enquue.pop())
        if not self._dequeue.empty():
            return self._dequeue.pop()
        raise IndexError("empty")
    def max(self):
        if not self._enquue.empty():
            return self._enquue.max() if self._dequeue.empty() else max(self._enquue.max(), self._dequeue.max())
        if not self._dequeue.empty():
            return self._dequeue.max()
        raise IndexError("empty")


