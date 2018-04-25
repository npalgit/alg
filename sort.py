def bubble_sort(arr):
    def swap(i,j):
        arr[i],arr[j]=arr[j],arr[i]
    n=len(arr)
    swapped = True
    while swapped:
        swapped = False
        for i in range(1,n):
            if arr[i-1]>arr[i]:
                swap(i-1,i)
                swapped = True


def comb_sort(arr):
    def swap(i,j):
        arr[i],arr[j]=arr[j],arr[i]
    n=len(arr)
    gap=n
    shrink=1.3
    sorted=False
    while not sorted:
        gap=int(gap//shrink)
        if gap>1:
            sorted=False
        else:
            gap=1
            sorted=True
        i=0
        while i+gap<n:
            if arr[i]>arr[i+gap]:
                swap(i,i+gap)
                sorted=False
            i=i+1

def counting_sort(arr):
    m=min(arr)
    different =0
    if m<0:
        different =-m
        for i in range(len(arr)):
            arr[i]+=-m
    k=max(arr)
    temp_arr =[0]*(k+1)
    for i in range(0,len(arr)):
        temp_arr[arr[i]]=temp_arr[arr[i]]+1
    for i in range(1,k+1):
        temp_arr[i]=temp_arr[i]+temp_arr[i-1]
    result_arr=[0]*len(arr)
    for i in range(len(arr)-1,-1,-1):
         result_arr[temp_arr[arr[i]]-1]=arr[i]-different
         temp_arr[arr[i]]=temp_arr[arr[i]]-1

    return result_arr


def max_heap_sort(arr):
    for i in range(len(arr)-1,0,-1):
        max_heapify(arr,i)
        temp =arr[0]
        arr[0]=arr[i]
        arr[i]=temp

def max_heapify(arr,end):
    last_parent =int((end-1)/2)

    for parent in range(last_parent,-1,-1):
        current_parent = parent

        while current_parent<=last_parent:
            child =2*current_parent+1
            if child+1 <= end and arr[child]<arr[child+1]:
                child = child +1
            if arr[child]>arr[current_parent]:
                temp=arr[current_parent]
                arr[current_parent]=arr[child]
                arr[child]=temp
                current_parent=child

            else:
                break

def min_heap_sort(arr):
    for i in range(0,len(arr)-1):
        min_heapify(arr,i)

def min_heapify(arr, start):
    end =len(arr)-1
    last_parent = int((end-start-1)/2)
    for parent in range(last_parent,-1,-1):
        current_parent =parent
        while current_parent<=last_parent:
            child = 2*current_parent+1
            if child+1 <=end-start and arr[child+start]>arr[child+1+start]:
                child= child +1
            if arr[child+start]<arr[current_parent+start]:
                temp =arr[current_parent+start]
                arr[current_parent+start]=arr[child+start]
                arr[child + start] =temp
                current_parent = child
            else:
                break
import timeit


def insertion_sort(arr):
    for i in range(len(arr)):
        cursor =arr[i]
        pos =i
        while pos > 0 and arr[pos-1]>cursor:
            arr[pos]=arr[pos-1]
            pos=pos-1
        arr[pos]=cursor
    return arr

def can_attend_meetings(intervals):
    intervals =sorted(intervals,key=lambda  x: x.start)
    for i in range(1,len(intervals)):
        if intervals[i].start < intervals[i-1].end:
            return False
    return True

def merge_sort(arr):
    if len(arr)<=1:
        return arr
    mid = len(arr)//2
    left,right =merge_sort(arr[mid:]), merge_sort(arr[:mid])
    return merge(left,right)
def merge(left,right):
    arr=[]
    left_cursor, right_cursor =0,0
    while left_cursor < len(left) and right_cursor < len(right):
        if left[left_cursor]<=right[right_cursor]:
            arr.append(left[left_cursor])
            left_cursor+=1
        else:
            arr.append(right[right_cursor])
            right_cursor+=1
    for i in range(left_cursor, len(left)):
        arr.append(left[i])
    for i in range(right_cursor,len(right)):
        arr.append(right[i])
    return arr


def quick_sort(arr,first ,last):
    if first<last:
        pos=partition(arr,first, last)
        print(arr[first:pos-1], arr[pos-1:last])
        quick_sort(arr,first, pos-1)
        quick_sort(arr,pos+1, last)
def partition(arr, first ,last):
    wall=first
    for pos in range(first, last):
        if arr[pos]<arr[last]:
            arr[pos],arr[wall]=arr[wall],arr[pos]
            wall+=1
    arr[wall],arr[last]=arr[last],arr[wall]
    print(wall)
    return wall

def selection_sort(arr):
    for i in range(len(arr)):
        minimum =i
        for j in range(i+1, len(arr)):
            if arr[j]<arr[minimum]:
                minimum =j
        arr[minimum],arr[i]=arr[i],arr[minimum]
    return arr


def sort_colors(nums):
    i=j=0
    for k in range(len(nums)):
        v=nums[k]
        nums[k]=2
        if v<2:
            nums[j]=1
            j+=1
        if v==0:
            nums[i]=0
            i+=1


def retDeps(visited, start):
    queue=[]
    out=[]
    queue.append(start)
    while queue:
        newNode = queue.pop()
        if newNode not in visited:
            visited.add(newNode)
        for child in depGraph[newNode]:
            queue.append(child)
            out.append(child)
    out.append(start)
    return out
def retDepGraph():
    visited=set()
    out=[]
    for pac in given:
        if pac in visited:
            continue
        visited.add(pac)
        if pac in depGraph:
            for child in depGraph[pac]:
                if child in visited:
                    continue
                out.extend(retDeps(visited,child))
        out.append(pac)
    print(out)

def wiggle_sort(nums):
    for i in range(len(nums)):
        if (i%2==1) ==(nums[i-1]>nums[i]):
            nums[i-1],nums[i]=nums[i],nums[i-1]

# @include
class Student(object):
    def __init__(self, name, grade_point_average):
        self.name = name
        self.grade_point_average = grade_point_average

    def __lt__(self, other):
        return self.name < other.name


students = [
    Student('A', 4.0), Student('C', 3.0), Student('B', 2.0), Student('D', 3.2)
]

# Sort according to __lt__ defined in Student. students remained unchanged.
students_sort_by_name = sorted(students)
# @exclude
assert all(a.name <= b.name
           for a, b in zip(students_sort_by_name, students_sort_by_name[1:]))
# @include

# Sort students in-place by grade_point_average.
students.sort(key=lambda student: student.grade_point_average)
# @exclude
assert all(a.grade_point_average <= b.grade_point_average
           for a, b in zip(students, students[1:]))

# @include
def intersect_two_sorted_arrays(A, B):
    i, j, intersection_A_B = 0, 0, []
    while i < len(A) and j < len(B):
        if A[i] == B[j]:
            if i == 0 or A[i] != A[i - 1]:
                intersection_A_B.append(A[i])
            i, j = i + 1, j + 1
        elif A[i] < B[j]:
            i += 1
        else:  # A[i] > B[j].
            j += 1
    return intersection_A_B
# @exclude

def merge_two_sorted_arrays(A, m, B, n):
    a, b, write_idx = m - 1, n - 1, m + n - 1
    while a >= 0 and b >= 0:
        if A[a] > B[b]:
            A[write_idx] = A[a]
            a -= 1
        else:
            A[write_idx] = B[b]
            b -= 1
        write_idx -= 1
    while b >= 0:
        A[write_idx] = B[b]
        write_idx, b = write_idx - 1, b - 1

class Name:

    def __init__(self, first_name, last_name):
        self.first_name, self.last_name = first_name, last_name

    def __eq__(self, other):
        return self.first_name == other.first_name

    def __lt__(self, other):
        return (self.first_name < other.first_name
                if self.first_name != other.first_name else
                self.last_name < other.last_name)
# @exclude

    def __repr__(self):
        return '%s %s' % (self.first_name, self.last_name)


# @include


def eliminate_duplicate(A):
    A.sort()  # Makes identical elements become neighbors.
    write_idx = 1
    for cand in A[1:]:
        if cand != A[write_idx - 1]:
            A[write_idx] = cand
            write_idx += 1
    del A[write_idx:]

def smallest_nonconstructible_value(A):
    max_constructible_value = 0
    for a in sorted(A):
        if a > max_constructible_value + 1:
            break
        max_constructible_value += a
    return max_constructible_value + 1

import collections
Event = collections.namedtuple('Event', ('start', 'finish'))

# Endpoint is a tuple (start_time, 0) or (end_time, 1) so that if times
# are equal, start_time comes first
Endpoint = collections.namedtuple('Endpoint', ('time', 'is_start'))


def find_max_simultaneous_events(A):
    # Builds an array of all endpoints.
    E = ([Endpoint(event.start, True) for event in A] +
         [Endpoint(event.finish, False) for event in A])
    # Sorts the endpoint array according to the time, breaking ties by putting
    # start times before end times.
    E.sort(key=lambda e: (e.time, not e.is_start))

    # Track the number of simultaneous events, record the maximum number of
    # simultaneous events.
    max_num_simultaneous_events, num_simultaneous_events = 0, 0
    for e in E:
        if e.is_start:
            num_simultaneous_events += 1
            max_num_simultaneous_events = max(num_simultaneous_events,
                                              max_num_simultaneous_events)
        else:
            num_simultaneous_events -= 1
    return max_num_simultaneous_events

Interval = collections.namedtuple('Interval', ('left', 'right'))


def add_interval(disjoint_intervals, new_interval):
    i, result = 0, []

    # Processes intervals in disjoint_intervals which come before new_interval.
    while (i < len(disjoint_intervals) and
           new_interval.left > disjoint_intervals[i].right):
        result.append(disjoint_intervals[i])
        i += 1

    # Processes intervals in disjoint_intervals which overlap with new_interval.
    while (i < len(disjoint_intervals) and
           new_interval.right >= disjoint_intervals[i].left):
        # If [a, b] and [c, d] overlap, union is [min(a, c), max(b, d)].
        new_interval = Interval(
            min(new_interval.left, disjoint_intervals[i].left),
            max(new_interval.right, disjoint_intervals[i].right))
        i += 1
    # Processes intervals in disjoint_intervals which come after new_interval.
    return result + [new_interval] + disjoint_intervals[i:]


Endpoint = collections.namedtuple('Endpoint', ('is_closed', 'val'))


class Interval:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __lt__(self, other):
        if self.left.val != other.left.val:
            return self.left.val < other.left.val
        # Left endpoints are equal, so now see if one is closed and the other open
        # - closed intervals should appear first.
        return self.left.is_closed and not other.left.is_closed


def union_of_intervals(intervals):
    # Empty input.
    if not intervals:
        return []

    # Sort intervals according to left endpoints of intervals.
    intervals.sort()
    result = [intervals[0]]
    for i in intervals:
        if intervals and (i.left.val < result[-1].right.val or
                          (i.left.val == result[-1].right.val and
                           (i.left.is_closed or result[-1].right.is_closed))):
            if (i.right.val > result[-1].right.val or
                (i.right.val == result[-1].right.val and i.right.is_closed)):
                result[-1].right = i.right
        else:
            result.append(i)
    return result

Person = collections.namedtuple('Person', ('age', 'name'))


def group_by_age(people):
    age_to_count = collections.Counter([person.age for person in people])
    age_to_offset, offset = {}, 0
    for age, count in age_to_count.items():
        age_to_offset[age] = offset
        offset += count

    while age_to_offset:
        from_age = next(iter(age_to_offset))
        from_idx = age_to_offset[from_age]
        to_age = people[from_idx].age
        to_idx = age_to_offset[people[from_idx].age]
        people[from_idx], people[to_idx] = people[to_idx], people[from_idx]
        # Use age_to_count to see when we are finished with a particular age.
        age_to_count[to_age] -= 1
        if age_to_count[to_age]:
            age_to_offset[to_age] = to_idx + 1
        else:
            del age_to_offset[to_age]


# @include
Player = collections.namedtuple('Player', ('height'))


class Team:
    def __init__(self, height):
        self._players = [Player(h) for h in height]

    # Checks if A can be placed in front of B.
    @staticmethod
    def valid_placement_exists(A, B):
        return all(a < b
                   for a, b in zip(sorted(A._players), sorted(B._players)))

def insertion_sort(L):
    dummy_head = ListNode(0, L)
    # The sublist consisting of nodes up to and including iter is sorted in
    # increasing order. We need to ensure that after we move to L.next this
    # property continues to hold. We do this by swapping L.next with its
    # predecessors in the list till it's in the right place.
    while L and L.next:
        if L.data > L.next.data:
            target, pre = L.next, dummy_head
            while pre.next.data < target.data:
                pre = pre.next
            temp, pre.next, L.next = pre.next, target, target.next
            target.next = temp
        else:
            L = L.next
    return dummy_head.next

from linkedlist import *
def stable_sort_list(L):
    # Base cases: L is empty or a single node, nothing to do.
    if not L or not L.next:
        return L

    # Find the midpoint of L using a slow and a fast pointer.
    pre_slow, slow, fast = None, L, L
    while fast and fast.next:
        pre_slow = slow
        fast, slow = fast.next.next, slow.next
    pre_slow.next = None  # Splits the list into two equal-sized lists.
    return merge_two_sorted_lists(stable_sort_list(L), stable_sort_list(slow))
# @exclude

def find_salary_cap(target_payroll, current_salaries):
    current_salaries.sort()
    unadjusted_salary_sum = 0.0
    for i, current_salary in enumerate(current_salaries):
        adjusted_people = len(current_salaries) - i
        adjusted_salary_sum = current_salary * adjusted_people
        if unadjusted_salary_sum + adjusted_salary_sum >= target_payroll:
            return (target_payroll - unadjusted_salary_sum) / adjusted_people
        unadjusted_salary_sum += current_salary
    # No solution, since target_payroll > existing payroll.
    return -1.0


