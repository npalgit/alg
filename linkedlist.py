class Node:
    def __init__(self, x=None):
        self.val = x
        self.next = None


def add_two_numbers(left, right):
    head = Node(0)
    current = head
    sum = 0
    while left or right:
        sum //= 10
        if left:
            sum += left.val
            left = left.next
        if right:
            sum += right.val
            right = right.next
        current.next = Node(sum % 10)
        current = current.next
    if sum // 10 == 1:
        current.next = Node(1)
    return head.next


class RandomListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


def copyRandomeList(head):
    dic = dict()
    m = n = head
    while m:
        dic[m] = RandomListNode(m.label)
        m = m.next
    while n:
        dic[n].next = dic.get(n.next)
        dic[n].random = dic.get(n.random)
        n = n.next
    return dic.get(head)


import collections


def copyRandomList2(head):
    copy = collections.defaultdict(lambda: RandomListNode(0))
    copy[None] = None
    node = head
    while node:
        copy[node].label = node.label
        copy[node].next = copy[node.next]
        copy[node].random = copy[node.random]
        node = node.next
    return copy[head]


def delete_node(node):
    node.val = node.next.val
    node.next = node.next.next


def is_cyclic(head):
    if not head:
        return False
    runner = head
    walker = head
    while runner.next and runner.next.next:
        runner = runner.next.next
        walker = walker.next
        if runner == walker:
            return True
    return False


def is_palindrome(head):
    if not head:
        return True
    fast, slow = head.next, head
    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next
    second = slow.next
    slow.next = None
    node = None
    while second:
        nxt = second.next
        second.next = node
        node = second
        second = nxt

    while node:
        if node.val != head.val:
            return False
        node = node.next
        head = head.next

    return True


def is_palindrome_stack(head):
    if not head or not head.next:
        return True

    slow = fast = cur = head
    while fast and fast.next:
        fast, slow = fast.next.next, slow.next

    stack = [slow.val]
    while slow.next:
        slow = slow.next
        stack.append(slow.val)

    while stack:
        if stack.pop() != cur.val:
            return False
        cur = cur.next
    return True


def removeDups(head):
    hashset = set()
    prev = Node()
    while head:
        if head.val in hashset:
            prev.next = head.next
        else:
            hashset.add(head.val)
            prev = head
        head = head.next


def removeDupsWithoutSet(head):
    current = head
    while current:
        runner = current
        while runner.next:
            if runner.next.val == current.val:
                runner.next = runner.next.next
            else:
                runner = runner.next
        current = current.next


def printLinkedList(head):
    string = ""
    while head.next:
        string += head.val + "->"
        head = head.next
    string += head.val
    print(string)


def reverse_list(head):
    if not head or not head.next:
        return head
    prev = None
    while head:
        current = head
        head = head.next
        current.next = prev
        prev = current
    return prev


def reverse_list_recursive(head):
    if head is None or head.next is None:
        return head
    p = head.next
    head.next = None
    reverse = reversed(p)
    p.next = head
    return reverse


def rotate_right(head, k):
    if not head or not head.next:
        return head
    current = head
    length = 1
    while current.next:
        current = current.next
        length += 1

    current.next = head
    k = k % length
    for i in range(length - k):
        current = current.next
    head = current.next
    current.next = None
    return head


def swap_pairs(head):
    if not head:
        return head
    start = Node()
    pre = start
    pre.next = head
    while pre.next and pre.next.next:
        a = pre.next
        b = pre.next.next
        pre.next, a.next, b.next = b, b.next, a
        pre = a
    return start.next


####################################################################
class ListNode:
    def __init__(self, data=0, next_node=None):
        self.data = data
        self.next = next_node

    def __repr__(self):
        return self.data, self.next_node


def search_list(L, key):
    while L and L.data != key:
        L = L.next
    return L


def insert_after(node, new_node):
    new_node.next = node.next
    node.next = new_node


def delete_after(node):
    node.next = node.next.next


def merge_two_sorted_lists(L1, L2):
    dummy_head = tail = ListNode()
    while L1 and L2:
        if L1.data < L2.data:
            tail.next, L1 = L1, L1.next
        else:
            tail.next, L2 = L2, L2.next
        tail = tail.next
    tail.next = L1 or L2
    return dummy_head.next


def reverse_sublist(L, start, finish):
    dummy_head = sublist_head = ListNode(0, L)
    for _ in range(1, start):
        sublist_head = sublist_head.next
    sublist_iter = sublist_head.next
    for _ in range(finish - start):
        temp = sublist_iter.next
        sublist_iter.next, temp.next, sublist_head.next = temp.next, sublist_head.next, temp
    return dummy_head.next


def has_cycle(head):
    fast = slow = head
    while fast and fast.next and fast.next.next:
        slow, fast = slow.next, fast.next.next
        if slow is fast:
            slow = head
            while slow is not fast:
                slow, fast = slow.next, fast.next
            return slow
    return None


def overlapping_no_cycle_lists(L1, L2):
    def length(L):
        length = 0
        while L:
            length += 1
            L = L.next
        return length

    L1_len, L2_len = length(L1), length(L2)
    if L1_len > L2_len:
        L1, L2 = L2, L1
    for _ in range(abs(L1_len - L2_len)):
        L2 = L2.next
    while L1 and L2 and L1 is not L2:
        L1, L2 = L1.next, L2.next
    return L1


def overlapping_lists(L1, L2):
    root1, root2 = has_cycle(L1), has_cycle(L2)
    if not root1 and not root2:
        return overlapping_no_cycle_lists(L1, L2)
    elif (root and not root2) or (not root1 and root2):
        return None
    temp = root2
    while True:
        temp = temp.next
        if temp is root1 or temp is root2:
            break
    if temp is not root1:
        return None

    def distance(a, b):
        dis = 0
        while a is not b:
            a = a.next
            dis += 1
        return dis

    stem1_length, stem2_length = distance(L1, root1), distance(L2, root2)
    if stem1_length > stem2_length:
        L2, L1 = L1, L2
        root1, root2 = root2, root1
    for _ in range(abs(stem1_length - stem2_length)):
        L2 = L2.next
        while L1 is not L2 and L1 is not root1 and L2 is not root2:
            L1, L2 = L1.next, L2.next
    return L1 if L1 is L2 else root1


def deletion_from_list(node_to_delete):
    node_to_delete.data = node_to_delete.next.data
    node_to_delete.next = node_to_delete.next.next


def remove_kth_last(L, k):
    dummy_head = ListNode(0, L)
    first = dummy_head.next
    for _ in range(k):
        first = first.next
    second = dummy_head
    while first:
        first, second = first.next, second.next
    second.next = second.next.next
    return dummy_head.next


def remove_duplicagtes(L):
    it = L
    while it:
        next_distinct = it.next
        while next_distinct and next_distinct.data == it.data:
            next_distinct = next_distinct.next
        it.next = next_distinct
        it = next_distinct
    return L


def cyclically_right_shift_list(L, k):
    if not L:
        return L
    tail, n = L, 1
    while tail.next:
        n += 1
        tail = tail.next
    k %= n
    if k == 0:
        return L
    tail.next = L
    steps_to_new_head, new_tail = n - k, tail
    while steps_to_new_head:
        steps_to_new_head -= 1
        new_tail = new_tail.next
    new_head = new_tail.next
    new_tail.next = None
    return new_head


def even_odd_merge_original(L):
    if not L:
        return L
    even_list_head = L
    even_list_iter, predecessor_even_list_iter = even_list_head, None
    odd_list_iter = odd_list_head = L.next
    while even_list_iter and odd_list_iter:
        even_list_iter.nexgt = odd_list_iter.next
        predecessor_even_list_iter = even_list_iter
        even_list_iter = even_list_iter.next
        if even_list_iter:
            odd_list_iter.next = even_list_iter.next
            odd_list_iter = odd_list_iter.next
    if even_list_iter:
        even_list_iter.next = odd_list_head
    else:
        predecessor_even_list_iter.next = odd_list_head
    return even_list_head


def is_linked_list_a_palindrom(L):
    slow = fast = L
    while fast and fast.next:
        fast, slow = fast.next.next, slow.next
    first_half_iter, second_half_iter = L, reverse_list(slow)
    while second_half_iter and first_half_iter:
        if second_half_iter.data != first_half_iter.data:
            return False
        second_half_iter, first_half_iter = second_half_iter.next, first_half_iter.next
    return True


def list_pivoting(L, x):
    less_head = less_iter = ListNode()
    equal_head = equal_iter = ListNode()
    greater_head = greater_iter = ListNode()
    while L:
        if L.data < x:
            less_iter.next = L
            less_iter = less_iter.next
        elif L.data == x:
            equal_iter.next = L
            equal_iter = equal_iter.next
        else:
            greater_iter.next = L
            greater_iter = greater_iter.next
        L = L.next
    greater_iter.next = None
    equal_iter.next = greater_head.next
    less_iter.next = equal_head.next
    return less_head.next


def add_two_numbers(L1, L2):
    place_iter = dummy_head = ListNode()
    carry = 0
    while L1 or L2 or carry:
        val = carry + (L1.data if L1 else 0) + (L2.data if L2 else 0)
        L1 = L1.next if L1 else None
        L2 = L2.next if L2 else None
        place_iter.next = ListNode(val % 10)
        carry, place_iter = val // 10, place_iter.next
    return dummy_head.next
