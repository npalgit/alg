def change_making(cents):
    COINS = [100, 50, 25, 10, 5, 1]
    num_coins = 0
    for coin in COINS:
        num_coins += cents / coin
        cents %= coin
    return num_coins
import collections
import sys
import random

# @include
PairedTasks = collections.namedtuple('PairedTasks', ('task_1', 'task_2'))


def optimum_task_assignment(task_durations):
    task_durations.sort()
    return [PairedTasks(task_durations[i], task_durations[~i])
            for i in range(len(task_durations) // 2)]

def minimum_total_waiting_time(service_times):
    # Sort the service times in increasing order.
    service_times.sort()
    total_waiting_time = 0
    for i, service_time in enumerate(service_times):
        num_remaining_queries = len(service_times) - (i + 1)
        total_waiting_time += service_time * num_remaining_queries
    return total_waiting_time

import operator

Interval = collections.namedtuple('Interval', ('left', 'right'))


def find_minimum_visits(intervals):
    if not intervals:
        return []

    # Sort intervals based on the right endpoints.
    intervals.sort(key=operator.attrgetter('right'))
    visits = []
    last_visit_time = float('-inf')
    for interval in intervals:
        if interval.left > last_visit_time:
            # The current right endpoint, last_visit_time, will not cover any
            # more intervals.
            last_visit_time = interval.right
            visits.append(last_visit_time)
    return visits


def has_two_sum(A, t):
    i, j = 0, len(A) - 1

    while i <= j:
        if A[i] + A[j] == t:
            return True
        elif A[i] + A[j] < t:
            i += 1
        else:  # A[i] + A[j] > t.
            j -= 1
    return False


def has_three_sum(A, t):
    A.sort()
    # Finds if the sum of two numbers in A equals to t - a.
    return any(has_two_sum(A, t - a) for a in A)

def majority_search(input_stream):
    candidate, candidate_count = None, 0
    for it in input_stream:
        if candidate_count == 0:
            candidate, candidate_count = it, candidate_count + 1
        elif candidate == it:
            candidate_count += 1
        else:
            candidate_count -= 1
    return candidate


MPG = 20


# gallons[i] is the amount of gas in city i, and distances[i] is the
# distance city i to the next city.
def find_ample_city(gallons, distances):
    remaining_gallons = 0
    CityAndRemainingGas = collections.namedtuple('CityAndRemainingGas',
                                                 ('city', 'remaining_gallons'))
    city_remaining_gallons_pair = CityAndRemainingGas(0, 0)
    num_cities = len(gallons)
    for i in range(1, num_cities):
        remaining_gallons += gallons[i - 1] - distances[i - 1] // MPG
        if remaining_gallons < city_remaining_gallons_pair.remaining_gallons:
            city_remaining_gallons_pair = CityAndRemainingGas(i,
                                                              remaining_gallons)
    return city_remaining_gallons_pair.city


def get_max_trapped_water(heights):
    i, j, max_water = 0, len(heights) - 1, 0
    while i < j:
        width = j - i
        max_water = max(max_water, width * min(heights[i], heights[j]))
        if heights[i] > heights[j]:
            j -= 1
        else:  # heights[i] <= heights[j].
            i += 1
    return max_water


def calculate_largest_rectangle_alternative(heights):
    # Calculate L.
    s = []
    L = []
    for i in range(len(heights)):
        while s and heights[s[-1]] >= heights[i]:
            del s[-1]
        L.append(-1 if not s else s[-1])
        s.append(i)

    # Clear stack for calculating R.
    s.clear()
    R = [None] * len(heights)
    for i in reversed(range(len(heights))):
        while s and heights[s[-1]] >= heights[i]:
            del s[-1]
        R[i] = len(heights) if not s else s[-1]
        s.append(i)

    # For each heights[i], find its maximum area include it.
    return max(heights[i] * (R[i] - L[i] - 1)
               for i in range(len(heights)) or [0])


# @include
def calculate_largest_rectangle(heights):
    pillar_indices, max_rectangle_area = [], 0
    # By appending [0] to heights, we can uniformly handle the computation for
    # rectangle area here.
    for i, h in enumerate(heights + [0]):
        while pillar_indices and heights[pillar_indices[-1]] >= h:
            height = heights[pillar_indices.pop()]
            width = i if not pillar_indices else i - pillar_indices[-1] - 1
            max_rectangle_area = max(max_rectangle_area, height * width)
        pillar_indices.append(i)
    return max_rectangle_area