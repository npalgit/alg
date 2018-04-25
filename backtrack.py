def all_perms2(elements):
    if len(elements) <= 1:
        return elements
    else:
        tmp = []
        for perm in all_perms2(elements[1:]): #insert A[0] into perm[1:]
            for i in range(len(elements)):
                tmp.append(perm[:i] + elements[0:1] + perm[i:])
        return tmp

def anagram(s1, s2):
    c1 = c2 = [0] * 26

    for i in range(len(s1)):
        pos = ord(s1[i]) - ord('a')
        c1[pos] +=1 # c -> count
    for i in range(len(s2)):
        pos = ord(s2[i]) - ord('a')
        c2[pos] +=1
    j = 0
    for j in range(26):
        if c1[j] != c2[j]: #check by c
            return False
    return True

#BT
def construct_candidates(constructed_sofar):
    global A, B, C
    array = A
    if 1 == len(constructed_sofar):
        array = B
    elif 2 == len(constructed_sofar):
        array = C
    return array


def over(constructed_sofar):
    global target
    sum = 0
    to_stop, reached_target = False, False
    for elem in constructed_sofar:
        sum += elem

    if sum >= target or len(constructed_sofar) >= 3:
        to_stop = True
        if sum == target and 3 == len(constructed_sofar):
            reached_target = True
    return to_stop, reached_target


def backtrack(constructed_sofar):
    to_stop, reached_target = over(constructed_sofar)
    if to_stop:
        if reached_target:
            print(constructed_sofar)
        return
    candidates = construct_candidates(constructed_sofar)
    for candidate in candidates:
        constructed_sofar.append(candidate)
        backtrack(constructed_sofar[:])
        constructed_sofar.pop()

#BT
def combinationSum(self, candidates, target):
    res = []
    candidates.sort()
    self.dfs(candidates, target, 0, [], res)
    return res


def dfs(self, nums, target, index, path, res):
    if target < 0: #exit 1 or 2
        return
    if target == 0:
        res.append(path)
        return

    for i in range(index, len(nums)):
        self.dfs(nums, target - nums[i], i, path + [nums[i]], res) # A, target ,  i, row, res


def add_operator(num, target):
    res = []
    if not num: return res
    helper(res, "", num, target, 0, 0, 0)
    return res


def helper(res, path, num, target, pos, prev, multed):
    if pos == len(num):
        if target == prev:
            res.append(path)
        return
    for i in range(pos, len(num)):
        if i != pos and num[pos] == '0':
            break
        cur = int(num[pos:i + 1])
        if pos == 0:
            helper(res, path + str(cur), num, target, i + 1, cur, cur)
        else:
            helper(res, path + "+" + str(cur), num, target, i + 1, prev + cur, cur)
            helper(res, path + "-" + str(cur), num, target, i + 1, prev - cur, cur)
            helper(res, path + "*" + str(cur), num, target, i + 1, prev - multed + multed * cur, multed * cur)


def getFactors(self, n):
    todo, combis = [(n, 2, [])], []
    while todo:
        n, i, combi = todo.pop()
        while i * i <= n:
            if n % i == 0:
                combis.append(combi + [i, n / i]),
                todo.append([n / i, i, combi + [i]])
            i += 1
    return combis


def getFactors(self, n):
    def factor(n, i, combi, combis):
        while i * i <= n:
            if n % i == 0:
                combis.append(combi + [i, n / i]),
                factor(n / i, i, combi + [i], combis)
            i += 1
        return combis

    return factor(n, 2, [], [])


def generate_abbreviations(word):
    result = []
    backtrack(result, word, 0, 0, "")
    return result


def backtrack(result, word, pos, count, cur):
    if pos == len(word):
        if count > 0:
            cur += str(count)
        result.append(cur)
        return

    if count > 0:
        backtrack(result, word, pos + 1, 0, cur + str(count) + word[pos])
    else:
        backtrack(result, word, pos + 1, 0, cur + word[pos])

    backtrack(result, word, pos + 1, count + 1, cur)


def gen_parenthesis(n):
    res = []
    add_pair(res, "", n, 0)
    return res


def add_pair(res, s, left, right):
    if left == 0 and right == 0:
        res.append(s)
        return
    if right > 0:
        add_pair(res, s + ")", left, right - 1)
    if left > 0:
        add_pair(res, s + "(", left - 1, right + 1)


def letter_combinations(digits):
    if digits == "":
        return []
    kmaps = {
        "2": "abc",
        "3": "def",
        "4": "ghi",
        "5": "jkl",
        "6": "mno",
        "7": "pqrs",
        "8": "tuv",
        "9": "wxyz"
    }

    ans = [""]
    for num in digits:
        tmp = []
        for an in ans:
            for char in kmaps[num]:
                tmp.append(an + char)
        ans = tmp

    return ans


digit_string = "23"
print(letter_combinations(digit_string))


def palindromic_substrings(s):
    if not s:
        return [[]]
    results = []
    for i in range(len(s), 0, -1):
        sub = s[:i]
        if sub == sub[::-1]:
            for rest in palindromic_substrings(s[i:]):
                results.append([sub] + rest)

    return results


def palindromic_substrings2(s):
    if not s:
        yield []
        return
    for i in range(len(s), 0, -1):
        sub = s[:i]
        if sub == sub[::-1]:
            for rest in palindromic_substrings2(s[i:]):
                yield [sub] + rest


def pattern_match(pattern, string):
    return backtrack2(pattern, string, {})


def backtrack2(pattern, string, dic):
    print(dic)
    if len(pattern) == 0 and len(string) > 0:
        return False

    if len(pattern) == len(string) == 0:
        return True

    for end in range(1, len(string) - len(pattern) + 2):
        if pattern[0] not in dic and string[:end] not in dic.values():
            dic[pattern[0]] = string[:end]
            if backtrack2(pattern[1:], string[end:], dic):
                return True
            del dic[pattern[0]]
        elif pattern[0] in dic and dic[pattern[0]] == string[:end]:
            if backtrack2(pattern[1:], string[end:], dic):
                return True

    return False


def permute(nums):
    perms = [[]]
    for n in nums:
        new_perms = []
        for perm in perms:
            for i in range(len(perm) + 1):
                new_perms.append(perm[:i] + [n] + perm[i:])
                print(i, perm[:i], [n], perm[i:], ">>>>", new_perms)
        perms = new_perms
    return perms



def permute_unique(nums):
    perms = [[]]
    for n in nums:
        new_perms = []
        for l in perms:
            for i in range(len(l) + 1):
                new_perms.append(l[:i] + [n] + l[i:])
                if i < len(l) and l[i] == n:
                    break
        perms = new_perms
    return perms


def subsets(nums):
    res = []
    backtrack3(res, nums, [], 0)
    return res


def backtrack3(res, nums, stack, pos):
    if pos == len(nums):
        res.append(list(stack))
    else:
        stack.append(nums[pos])
        backtrack3(res, nums, stack, pos + 1)
        stack.pop()
        backtrack3(res, nums, stack, pos + 1)


def subsets2(self, nums):
    res = [[]]
    for num in sorted(nums):
        res += [item + [num] for item in res]


def subsets_unique(nums):
    res = set()
    backtrack4(res, nums, [], 0)
    return list(res)


def backtrack4(res, nums, stack, pos):
    if pos == len(nums):
        res.add(tuple(stack))
    else:
        stack.append(nums[pos])
        backtrack4(res, nums, stack, pos + 1)
        stack.pop()
        backtrack4(res, nums, stack, pos + 1)

def find_words(board, words):
    trie = {}
    for word in words:
        curr_trie = trie
        for char in word:
            if char not in curr_trie:
                curr_trie[char] = {}
            curr_trie = curr_trie[char]
        curr_trie['#'] = '#'
    result = set()
    used = [[False] * len(board[0]) for _ in range(len(board))]
    for i in range(len(board)):
        for j in range(len(board[0])):
            backtrack5(board, i, j, trie, '', used, result)
    return list(result)


def backtrack5(board, i, j, trie, pre, used, result):
    if '#' in trie:
        result.add(pre)
    if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]):
        return
    if not used[i][j] and board[i][j] in trie:
        used[i][j] = True
        backtrack5(board, i + 1, j, trie[board[i][j]], pre + board[i][j], used, result)
        backtrack5(board, i, j + 1, trie[board[i][j]], pre + board[i][j], used, result)
        backtrack5(board, i - 1, j, trie[board[i][j]], pre + board[i][j], used, result)
        backtrack5(board, i, j - 1, trie[board[i][j]], pre + board[i][j], used, result)
        used[i][j] = False

# @include
def gcd(x, y):
    return x if y == 0 else gcd(y, x % y)
# @exclude

num_steps = 0


# @include
def compute_tower_hanoi(num_rings):
    def compute_tower_hanoi_steps(num_rings_to_move, from_peg, to_peg, use_peg):
        # @exclude
        global num_steps
        # @include
        if num_rings_to_move > 0:
            compute_tower_hanoi_steps(
                num_rings_to_move - 1, from_peg, use_peg, to_peg)
            pegs[to_peg].append(pegs[from_peg].pop())
            print('Move from peg', from_peg, 'to peg', to_peg)
            # @exclude
            num_steps += 1
            # @include
            compute_tower_hanoi_steps(
                num_rings_to_move - 1, use_peg, to_peg, from_peg)

    # Initialize pegs.
    NUM_PEGS = 3
    pegs = [list(reversed(range(1, num_rings + 1)))] + [[]
                                                        for _ in range(1, NUM_PEGS)]
    compute_tower_hanoi_steps(num_rings, 0, 1, 2)


def n_queens(n):
    def solve_n_queens(row):
        if row == n:
            # All queens are legally placed.
            result.append(list(col_placement))
            return
        for col in range(n):
            # Test if a newly placed queen will conflict any earlier queens
            # placed before.
            if all(abs(c - col) not in (0, row - i)
                   for i, c in enumerate(col_placement[:row])):
                col_placement[row] = col
                solve_n_queens(row + 1)

    result, col_placement = [], [0] * n
    solve_n_queens(0)
    return result
# @exclude
def permutations(A):
    def directed_permutations(i):
        if i == len(A) - 1:
            result.append(A.copy())
            return

        # Try every possibility for A[i].
        for j in range(i, len(A)):
            A[i], A[j] = A[j], A[i]
            # Generate all permutations for A[i + 1:].
            directed_permutations(i + 1)
            A[i], A[j] = A[j], A[i]

    result = []
    directed_permutations(0)
    return result

def generate_power_set(input_set):
    # Generate all subsets whose intersection with input_set[0], ...,
    # input_set[to_be_selected - 1] is exactly selected_so_far.
    def directed_power_set(to_be_selected, selected_so_far):
        if to_be_selected == len(input_set):
            power_set.append(list(selected_so_far))
            return

        directed_power_set(to_be_selected + 1, selected_so_far)
        # Generate all subsets that contain input_set[to_be_selected].
        directed_power_set(to_be_selected + 1,
                           selected_so_far + [input_set[to_be_selected]])

    power_set = []
    directed_power_set(0, [])
    return power_set


def combinations(n, k):
    def directed_combinations(offset, partial_combination):
        if len(partial_combination) == k:
            result.append(list(partial_combination))
            return

        # Generate remaining combinations over {offset, ..., n - 1} of size
        # num_remaining.
        num_remaining = k - len(partial_combination)
        i = offset
        while i <= n and num_remaining <= n - i + 1:
            directed_combinations(i + 1, partial_combination + [i])
            i += 1

    result = []
    directed_combinations(1, [])
    return result

def generate_balanced_parentheses(num_pairs):
    def directed_generate_balanced_parentheses(num_left_parens_needed,
                                               num_right_parens_needed,
                                               valid_prefix,
                                               result=[]):
        if num_left_parens_needed > 0:  # Able to insert '('.
            directed_generate_balanced_parentheses(num_left_parens_needed - 1,
                                                   num_right_parens_needed,
                                                   valid_prefix + '(')
        if num_left_parens_needed < num_right_parens_needed:
            # Able to insert ')'.
            directed_generate_balanced_parentheses(num_left_parens_needed,
                                                   num_right_parens_needed - 1,
                                                   valid_prefix + ')')
        if not num_right_parens_needed:
            result.append(valid_prefix)
        return result

    return directed_generate_balanced_parentheses(num_pairs, num_pairs, '')



def palindrome_partitioning(input):
    def directed_palindrome_partitioning(offset, partial_partition):
        if offset == len(input):
            result.append(list(partial_partition))
            return

        for i in range(offset + 1, len(input) + 1):
            prefix = input[offset:i]
            if prefix == prefix[::-1]:
                directed_palindrome_partitioning(i,
                                                 partial_partition + [prefix])

    result = []
    directed_palindrome_partitioning(0, [])
    return result

from tree import BinaryTreeNode
def generate_all_binary_trees(num_nodes):
    if num_nodes == 0:  # Empty tree, add as a None.
        return [None]

    result = []
    for num_left_tree_nodes in range(num_nodes):
        num_right_tree_nodes = num_nodes - 1 - num_left_tree_nodes
        left_subtrees = generate_all_binary_trees(num_left_tree_nodes)
        right_subtrees = generate_all_binary_trees(num_right_tree_nodes)
        # Generates all combinations of left_subtrees and right_subtrees.
        result += [
            BinaryTreeNode(0, left, right)
            for left in left_subtrees for right in right_subtrees
        ]
    return result


def solve_sudoku(partial_assignment):
    def solve_partial_sudoku(i, j):
        if i == len(partial_assignment):
            i = 0  # Starts a row.
            j += 1
            if j == len(partial_assignment[i]):
                return True  # Entire matrix has been filled without conflict.

        # Skips nonempty entries.
        if partial_assignment[i][j] != EMPTY_ENTRY:
            return solve_partial_sudoku(i + 1, j)

        def valid_to_add(i, j, val):
            # Check row constraints.
            if any(val == partial_assignment[k][j]
                   for k in range(len(partial_assignment))):
                return False

            # Check column constraints.
            if val in partial_assignment[i]:
                return False

            # Check region constraints.
            region_size = int(math.sqrt(len(partial_assignment)))
            I = i // region_size
            J = j // region_size
            return not any(
                val == partial_assignment[region_size * I +
                                          a][region_size * J + b]
                for a, b in itertools.product(range(region_size), repeat=2))

        for val in range(1, len(partial_assignment) + 1):
            # It's substantially quicker to check if entry val with any of the
            # constraints if we add it at (i,j) adding it, rather than adding it and
            # then checking all constraints. The reason is that we know we are
            # starting with a valid configuration, and the only entry which can
            # cause a problem is entry val at (i,j).
            if valid_to_add(i, j, val):
                partial_assignment[i][j] = val
                if solve_partial_sudoku(i + 1, j):
                    return True
        partial_assignment[i][j] = EMPTY_ENTRY  # Undo assignment.
        return False

    EMPTY_ENTRY = 0
    return solve_partial_sudoku(0, 0)

def gray_code(num_bits):
    def directed_gray_code(history):
        def differs_by_one_bit(x, y):
            bit_difference = x ^ y
            return bit_difference and not (bit_difference &
                                           (bit_difference - 1))

        if len(result) == 1 << num_bits:
            # Check if the first and last codes differ by one bit.
            return differs_by_one_bit(result[0], result[-1])

        for i in range(num_bits):
            previous_code = result[-1]
            candidate_next_code = previous_code ^ (1 << i)
            if candidate_next_code not in history:
                history.add(candidate_next_code)
                result.append(candidate_next_code)
                if directed_gray_code(history):
                    return True
                history.remove(candidate_next_code)
                del result[-1]
        return False

    result = [0]
    directed_gray_code(set([0]))
    return result


import collections
Edge = collections.namedtuple('Edge', ('root', 'length'))


def compute_diameter(T):
    HeightAndDiameter = collections.namedtuple('HeightAndDiameter',
                                               ('height', 'diameter'))

    def compute_height_and_diameter(r):
        diameter = float('-inf')
        heights = [0.0, 0.0]  # Stores the max two heights.
        for e in r.edges:
            h_d = compute_height_and_diameter(e.root)
            if h_d.height + e.length > heights[0]:
                heights = [h_d.height + e.length, heights[0]]
            elif h_d.height + e.length > heights[1]:
                heights[1] = h_d.height + e.length
            diameter = max(diameter, h_d.diameter)
        return HeightAndDiameter(heights[0],
                                 max(diameter, heights[0] + heights[1]))

    return compute_height_and_diameter(T).diameter if T else 0.0