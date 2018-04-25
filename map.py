def HashTable(object):
    _empty=object()
    _deleted=object()

    def __init__(self, size=11):
        self.size = size
        self._len=0
        self._keys=[self._empty]*size
        self._values = [self._empty]*size

    def put(self, key, value):
        initial_hash = hash_ = self.hash(key)

        while True:
            if self._keys[hash_] is self._empty or self._keys[hash_] is self._deleted:
                self._keys[hash_] = key
                self._values[hash_] = value
                self._len +=1
                return
            elif self._key[hash_] ==key:
                self._keys[hash_] =key
                self._values[hash_] =value
                return
            hash_ = self._refresh(hash_)

            if initial_hash == hash_:
                raise ValueError("Table is full")

    def get(self,key):
        initial_hash = hash_ = self.hash(key)
        while True:
            if self._keys[hash_] is self._empty:
                return None
            elif self._keys[hash_] == key:
                return self._values[hash_]
            hash_ = self._rehash(hash_)
            if initial_hash == hash_:
                return None
    def del_(self,key):
        initial_hash =hash_ = self.hash(key)
        while True:
            if self._keys[hash_] is self._empty:
                return None
            elif self._keys[hahs_]==key:
                self._keys[hash_] = self._deleted
                self._values[hash_] = self._deleted
                self._len -=1
                return
            hash_ = self.rehash(hash_)
            if initial_hash ==hash_:
                return None
    def hash(self, key):
        return key%self.size
    def _rehash(self, old_hash):
        return (old_hash+1) % self.size
    def __getitem__(self,key):
        return self.get(key)
    def __delitem__(self,key):
        return self.del_(key)
    def __setitem__(self,key,value):
        self.put(key,value)
    def __len__(self):
        return self._len

    def ResizableHashTable(HashTable):
        MIN_SIZE =8
        def __init__(self):
            super().__init__(self.MIN_SIZE)

        def put(self,key,value):
            rv=super().put(key,value)
            if len(self)>=(self.size*2)/3:
                self.__resize()
        def __resize(self):
            keys,values=self._keys,self._values
            self.size *=2
            self._len =0
            self._keys = [self._empty] *self.size
            self._values = [self._empty]*self.size
            for key, value in zip(keys,values):
                if key is not self._empty and key is not self._deleted:
                    self.put(key,value)

def max_common_sub_string(s1,s2):
    s2dic ={s2[i]: i for i in range(len(s2))}
    maxr=0
    subs=''
    i=0
    while i<len(s1):
        if s1[i] in s2dic:
            j=s2dic[s1[i]]
            k=i
            while j<len(s2) and k<len(s1) and s1[k]==s2[j]:
                k+=1
                j+=1
            if k-i>maxr:
                maxr=k-i
                subs=s1[i:k]
            i=k
        else:
            i+=1
    return subs

import random

class RandomizedSet:
    def __init__(self):
        self.nums=[]
        self.idxs={}

    def insert(self,val):
        if val not in self.idxs:
            self.nums.append(val)
            self.idxs[val]=len(self.nums)-1
            return True
        return False

    def remove(self,val):
        if val in self.idxs:
            idx,last =self.idxs[val], self.nums[-1]
            self.nums[idx], self.idxs[last]=last,idx
            self.nums.pop()
            self.idxs.pop(val,0)
            return True
        return False

    def get_random(self):
        idx=random.randint(0,len(self.nums)-1)
        return self.nums[idx]


def is_valid_sudoku(self, board):
    seen=[]
    for i, row in enumerate(board):
        for j, c in enumerate(row):
            if c!='.':
                seen+=[(c,j),(i,c),(i/3,j/3,c)]
    return len(seen)==len(set(seen))



##########################################################################
import functools
import sys
import string
import random
import collections

def string_hash(s, modulus):
    MULT = 997
    return functools.reduce(lambda v, c: (v * MULT + ord(c)) % modulus, s, 0)

def find_anagrams(dictionary):
    sorted_string_to_anagrams = collections.defaultdict(list)
    for s in dictionary:
        # Sorts the string, uses it as a key, and then appends the original
        # string as another value into hash table.
        sorted_string_to_anagrams[''.join(sorted(s))].append(s)

    return [
        group for group in sorted_string_to_anagrams.values() if len(group) >= 2
    ]

class ContactList:

    def __init__(self, names):
        '''
        names is a list of strings.
        '''
        self.names = names

    def __hash__(self):
        # Conceptually we want to hash the set of names. Since the set type is
        # mutable, it cannot be hashed. Therefore we use frozenset.
        return hash(frozenset(self.names))

    def __eq__(self, other):
        return set(self.names) == set(other.names)


def merge_contact_lists(contacts):
    '''
    contacts is a list of ContactList.
    '''
    return list(set(contacts))

import collections
# @include
c = collections.Counter(a=3, b=1)
d = collections.Counter(a=1, b=2)
# add two counters together:  c[x] + d[x], collections.Counter({'a': 4, 'b': 3})
c + d
# subtract (keeping only positive counts), collections.Counter({'a': 2})
c - d
# intersection:  min(c[x], d[x]), collections.Counter({'a': 1, 'b': 1})
c & d
# union:  max(c[x], d[x]), collections.Counter({'a': 3, 'b': 2})
c | d
# @exclude

def can_string_be_a_palindrome(s):
    # A string can be permuted to form a palindrome if and only if the number
    # of chars whose frequencies is odd is at most 1.
    return sum(v % 2 for v in collections.Counter(s).values()) <= 1

def is_letter_constructible_from_magazine(letter_text, magazine_text):
    # Compute the frequencies for all chars in letter_text.
    char_frequency_for_letter = collections.Counter(letter_text)

    # Checks if characters in magazine_text can cover characters in
    # char_frequency_for_letter.
    for c in magazine_text:
        if c in char_frequency_for_letter:
            char_frequency_for_letter[c] -= 1
            if char_frequency_for_letter[c] == 0:
                del char_frequency_for_letter[c]
                if not char_frequency_for_letter:
                    # All characters for letter_text are matched.
                    return True

    # Empty char_frequency_for_letter means every char in letter_text can be
    # covered by a character in magazine_text.
    return not char_frequency_for_letter

class LRUCache:

    def __init__(self, capacity):
        self._isbn_price_table = collections.OrderedDict()
        self._capacity = capacity

    def lookup(self, isbn):
        if isbn not in self._isbn_price_table:
            return False, None
        price = self._isbn_price_table.pop(isbn)
        self._isbn_price_table[isbn] = price
        return True, price

    def insert(self, isbn, price):
        # We add the value for key only if key is not present - we don't update
        # existing values.
        if isbn in self._isbn_price_table:
            price = self._isbn_price_table.pop(isbn)
        elif self._capacity <= len(self._isbn_price_table):
            self._isbn_price_table.popitem(last=False)
        self._isbn_price_table[isbn] = price

    def erase(self, isbn):
        return self._isbn_price_table.pop(isbn, None) is not None
# @exclude

def lca(node_0, node_1):
    iter_0, iter_1 = node_0, node_1
    nodes_on_path_to_root = set()
    while iter_0 or iter_1:
        # Ascend tree in tandem for these two nodes.
        if iter_0:
            if iter_0 in nodes_on_path_to_root:
                return iter_0
            nodes_on_path_to_root.add(iter_0)
            iter_0 = iter_0.parent
        if iter_1:
            if iter_1 in nodes_on_path_to_root:
                return iter_1
            nodes_on_path_to_root.add(iter_1)
            iter_1 = iter_1.parent
    raise ValueError('node_0 and node_1 are not in the same tree')
# @exclude

def find_nearest_repetition(paragraph):
    word_to_latest_index, nearest_repeated_distance = {}, float('inf')
    for i, word in enumerate(paragraph):
        if word in word_to_latest_index:
            latest_equal_word = word_to_latest_index[word]
            nearest_repeated_distance = min(nearest_repeated_distance,
                                            i - latest_equal_word)
        word_to_latest_index[word] = i
    return nearest_repeated_distance

def find_smallest_subarray_covering_set(paragraph, keywords):
    keywords_to_cover = collections.Counter(keywords)
    result = (-1, -1)
    remaining_to_cover = len(keywords)
    left = 0
    for right, p in enumerate(paragraph):
        if p in keywords:
            keywords_to_cover[p] -= 1
            if keywords_to_cover[p] >= 0:
                remaining_to_cover -= 1

        # Keeps advancing left until keywords_to_cover does not contain all
        # keywords.
        while remaining_to_cover == 0:
            if result == (-1, -1) or right - left < result[1] - result[0]:
                result = (left, right)
            pl = paragraph[left]
            if pl in keywords:
                keywords_to_cover[pl] += 1
                if keywords_to_cover[pl] > 0:
                    remaining_to_cover += 1
            left += 1
    return result

def find_smallest_subarray_covering_subset(stream, query_strings):
    class DoublyLinkedListNode:

        def __init__(self, data=None):
            self.data = data
            self.next = self.prev = None

    class LinkedList:

        def __init__(self):
            self.head = self.tail = None
            self._size = 0

        def __len__(self):
            return self._size

        def insert_after(self, value):
            node = DoublyLinkedListNode(value)
            node.prev = self.tail
            if self.tail:
                self.tail.next = node
            else:
                self.head = node
            self.tail = node
            self._size += 1

        def remove(self, node):
            if node.next:
                node.next.prev = node.prev
            else:
                self.tail = node.prev
            if node.prev:
                node.prev.next = node.next
            else:
                self.head = node.next
            node.next = node.prev = None
            self._size -= 1

    # Tracks the last occurrence (index) of each string in query_strings.
    loc = LinkedList()
    d = {s: None for s in query_strings}
    res = (-1, -1)
    for idx, s in enumerate(stream):
        if s in d:  # s is in query_strings.
            it = d[s]
            if it is not None:
                # Explicitly remove s so that when we add it, it's the string most
                # recently added to loc.
                loc.remove(it)
            loc.insert_after(idx)
            d[s] = loc.tail

            if len(loc) == len(query_strings):
                # We have seen all strings in query_strings, let's get to work.
                if res == (-1, -1) or idx - loc.head.data < res[1] - res[0]:
                    res = (loc.head.data, idx)
    return res

Subarray = collections.namedtuple('Subarray', ('start', 'end'))


def find_smallest_sequentially_covering_subset(paragraph, keywords):
    # Maps each keyword to its index in the keywords array.
    keyword_to_idx = {k: i for i, k in enumerate(keywords)}

    # Since keywords are uniquely identified by their indices in keywords
    # array, we can use those indices as keys to lookup in an array.
    latest_occurrence = [-1] * len(keywords)
    # For each keyword (identified by its index in keywords array), the length
    # of the shortest subarray ending at the most recent occurrence of that
    # keyword that sequentially cover all keywords up to that keyword.
    shortest_subarray_length = [float('inf')] * len(keywords)

    shortest_distance = float('inf')
    result = Subarray(-1, -1)
    for i, p in enumerate(paragraph):
        if p in keyword_to_idx:
            keyword_idx = keyword_to_idx[p]
            if keyword_idx == 0:  # First keyword.
                shortest_subarray_length[keyword_idx] = 1
            elif shortest_subarray_length[keyword_idx - 1] != float('inf'):
                distance_to_previous_keyword = (
                    i - latest_occurrence[keyword_idx - 1])
                shortest_subarray_length[keyword_idx] = (
                    distance_to_previous_keyword +
                    shortest_subarray_length[keyword_idx - 1])
            latest_occurrence[keyword_idx] = i

            # Last keyword, for improved subarray.
            if (keyword_idx == len(keywords) - 1 and
                    shortest_subarray_length[-1] < shortest_distance):
                shortest_distance = shortest_subarray_length[-1]
                result = Subarray(i - shortest_distance + 1, i)
    return result

def longest_subarray_with_distinct_entries(A):
    # Records the most recent occurrences of each entry.
    most_recent_occurrence = {}
    longest_dup_free_subarray_start_idx = result = 0
    for i, a in enumerate(A):
        # Defer updating dup_idx until we see a duplicate.
        if a in most_recent_occurrence:
            dup_idx = most_recent_occurrence[a]
            # A[i] appeared before. Did it appear in the longest current
            # subarray?
            if dup_idx >= longest_dup_free_subarray_start_idx:
                result = max(result, i - longest_dup_free_subarray_start_idx)
                longest_dup_free_subarray_start_idx = dup_idx + 1
        most_recent_occurrence[a] = i
    return max(result, len(A) - longest_dup_free_subarray_start_idx)



def find_longest_contained_range(A):
    if not A:
        return 0

    t = set()  # records the unique appearance.
    # L[i] stores the upper range for i.
    L = {}
    # U[i] stores the lower range for i.
    U = {}
    for a in A:
        if a not in t:
            t.add(a)
            L[a] = U[a] = a
            # Merges with the interval starting on A[i] + 1.
            if a + 1 in L:
                L[U[a]] = L[a + 1]
                U[L[a + 1]] = U[a]
                del L[a + 1]
                del U[a]
            # Merges with the interval ending on A[i] - 1.
            if a - 1 in U:
                U[L[a]] = U[a - 1]
                L[U[a - 1]] = L[a]
                del U[a - 1]
                del L[a]

    m = max(L.items(), key=lambda a: a[1] - a[0])
    return m[1] - m[0] + 1


# @include
def longest_contained_range(A):
    # unprocessed_entries records the existence of each entry in A.
    unprocessed_entries = set(A)

    max_interval_size = 0
    while unprocessed_entries:
        a = unprocessed_entries.pop()

        # Finds the lower bound of the largest range containing a.
        lower_bound = a - 1
        while lower_bound in unprocessed_entries:
            unprocessed_entries.remove(lower_bound)
            lower_bound -= 1

        # Finds the upper bound of the largest range containing a.
        upper_bound = a + 1
        while upper_bound in unprocessed_entries:
            unprocessed_entries.remove(upper_bound)
            upper_bound += 1

        max_interval_size = max(max_interval_size,
                                upper_bound - lower_bound - 1)
    return max_interval_size

import operator
import heapq
def find_student_with_highest_best_of_three_scores(name_score_data):
    student_scores = collections.defaultdict(list)
    for line in name_score_data:
        name, score = line.split()
        if len(student_scores[name]) < 3:
            heapq.heappush(student_scores[name], int(score))
        else:
            heapq.heappushpop(student_scores[name], int(score))
    return max([(sum(scores), name) for name, scores in student_scores.items()
                if len(scores) == 3],
               key=operator.itemgetter(0),
               default='no such student')[1]


def find_all_substrings(s, words):
    def match_all_words_in_dict(start):
        curr_string_to_freq = collections.Counter()
        for i in range(start, start + len(words) * unit_size, unit_size):
            curr_word = s[i:i + unit_size]
            it = word_to_freq[curr_word]
            if it == 0:
                return False
            curr_string_to_freq[curr_word] += 1
            if curr_string_to_freq[curr_word] > it:
                # curr_word occurs too many times for a match to be possible.
                return False
        return True

    word_to_freq = collections.Counter(words)
    unit_size = len(words[0])
    return [
        i for i in range(len(s) - unit_size * len(words) + 1)
        if match_all_words_in_dict(i)
    ]


def test_collatz_conjecture(n):
    # Stores odd numbers already tested to converge to 1.
    verified_numbers = set()

    # Starts from 3, hypothesis holds trivially for 1.
    for i in range(3, n + 1):
        sequence = set()
        test_i = i
        while test_i >= i:
            if test_i in sequence:
                # We previously encountered test_i, so the Collatz sequence has
                # fallen into a loop. This disproves the hypothesis, so we
                # short-circuit, returning False.
                return False
            sequence.add(test_i)

            if test_i % 2:  # Odd number.
                if test_i in verified_numbers:
                    break  # test_i has already been verified to converge to 1.
                verified_numbers.add(test_i)
                test_i = 3 * test_i + 1  # Multiply by 3 and add 1.
            else:
                test_i //= 2  # Even number, halve it.
    return True