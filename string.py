def add_binary(a, b):
    s = ""
    c, i, j = 0, len(a) - 1, len(b) - 1
    zero = ord('0')
    while i >= 0 or j >= 0 or c == 1:
        if i >= 0:
            c += ord(a[i]) - zero
            i -= 1
        if j >= 0:
            c += ord(b[j]) - zero
            j -= 1
        s = chr(c % 2 + zero) + s
        c /= 2
    return s



def match_symbol(words, symbols):
    import re
    combined = []
    for s in symbols:
        for c in words:
            r = re.search(s, c)
            if r:
                combined.append(re.sub(s, "[{}]".format(s), c))
    return combined


def match_symbol2(words, symbols):
    res = []
    symbols = sorted(symbols, key=lambda _: len(_), reverse=True)
    for word in words:
        for symbol in symbols:
            word_repaced = ''
            if word.find(symbol) != -1:
                word_repaced = word.repace(symbol, '[' + symbol + ']')
                res.append(word_repaced)
                break
        if word_repaced == '':
            res.append(word)
    return res


from functools import reduce


class TrieNode:
    def __init__(self):
        self.c = dict()
        self.sym = None


def bracket(words, symbols):
    root = TrieNode()
    for s in symbols:
        t = root
        for char in s:
            if char not in t.c:
                t.c[char] = TrieNode()
            t = t.c[char]
        t.sym = s
    result = dict()
    for word in words:
        i = 0
        symlist = list()
        while i < len(word):
            j, t = i, root
            while j < len(word) and word[j] in t.c:
                t = t.c[word[j]]
                if t.sym is not None:
                    symlist.append((j + 1 - len(t.sym), j + 1, t.sym))
                j += 1
            i += 1
        if len(symlist) > 0:
            sym = reduce(lambda x, y: x if x[1] - x[0] >= y[1] - y[0] else y, symlist)
            result[word] = "{}[{}]{}".format(word[:sym[0]], sym[2], word[sym[1]:])
    return tuple(word if word not in result else result[word] for word in words)


def decode_string(s):
    stack, cur_num, cur_string = [], 0, ''
    for c in s:
        if c == '[':
            stack.append((cur_string, cur_num))
            cur_string = ''
            cur_num = 0
        elif c == ']':
            prev_string, num = stack.pop()
            cur_string = prev_string + num * cur_string
        elif c.isdigit():
            cur_num = cur_num * 10 + int(c)
        else:
            cur_string += c
    return cur_string


def encode(strs):
    res = ''
    for string in strs.split():
        res += str(len(string)) + ":" + string
    return res


def decode(s):
    strs = []
    i = 0
    while i < len(s):
        index = s.find(":", i)
        size = int(s[i:index])
        strs.append(s[index + 1:index + 1 + size])
        i = index + 1 + size
    return strs


def groupAnagrams(strs):
    d = {}
    ans = []
    k = 0
    for str in strs:
        sstr = ''.join(sorted(str))
        if sstr not in d:
            d[sstr] = k
            k = k + 1
            ans.append([])
            ans[-1].append(str)
        else:
            ans[d[sstr]].append(str)
    return ans


def int_to_roman(num):
    M = ["", "M", "MM", "MMM"];
    C = ["", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"];
    X = ["", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"];
    I = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"];
    return M[num // 1000] + C[num % 1000 // 100] + X[num % 100 // 10] + I[num % 10]

def is_palindrome(s):
    i = 0
    j = len(s) - 1
    while i < j:
        while i < j and not s[i].isalnum():
            i += 1
        while i < j and not s[j].isalnum():
            j -= 1
        if s[i].lower() != s[j].lower():
            return False
        i, j = i + 1, j - 1
    return True


def license_number(key, k):
    res, alnum = [], []
    for char in key:
        if char != '_':
            alnum.append(char)
    for i, char in enumerate(reversed(alnum)):
        res.append(char)
        if (i + 1) % k == 0 and i != len(alnum) - 1:
            res.append("-")
    return "".join(res[::-1])



def make_sentence(str_piece, dictionarys):
    global count
    if len(str_piece) == 0:
        return True
    for i in range(0, len(str_piece)):
        prefix, suffix = str_piece[0:i], str_piece[i:]
        if (prefix in dictionarys and suffix in dictionarys) or (
                prefix in dictionarys and make_sentence(suffix, dictionarys)):
            count += 1
    return True



def multiply(num1, num2):
    carry = 1
    interm = []
    zero = ord('0')
    i_pos = 1
    for i in reversed(num1):
        j_pos = 1
        add = 0
        for j in reversed(num2):
            mult = (ord(i) - zero) * (ord(j) - zero) * j_pos * i_pos
            j_pos *= 10
            add += mult
        i_pos *= 10
        interm.append(add)
    return str(sum(interm))


class RollingHash:
    def __int__(self, text, sizeWord):
        self.text = text
        self.hash = 0
        self.sizeWord = sizeWord

        for i in range(0, sizeWord):
            self.hash += (ord(self.text[i]) - ord("a") + 1) * (26 ** (sizeWord - i - 1))
        self.window_start = 0
        self.window_end = sizeWord

    def move_window(self):
        if self.window_end <= len(self.text) - 1:
            self.hash -= (ord(self.text[self.window_start]) - ord("a") + 1) * 26 ** (self.sizeWord - 1)
            self.hash *= 26
            self.hash += ord(self.text[self.window_end] - ord("a") + 1)
            self.window_start += 1
            self.window_end += 1

    def window_text(self):
        return self.text[self.window_start:self.window_end]


def rabin_karp(word, text):
    if word == "" or text == "":
        return None
    if len(word) > len(text):
        return None

    rolling_hash = RollingHash(text, len(word))
    word_hash = RollingHash(word, len(word))
    for i in range(len(text) - len(word) + 1):
        if rolling_hash.hash == word_hash.hash:
            if rolling_hash.window_text() == word:
                return i
            rolling_hash.move_window()
    return None


def recursive(s):
    l = len(s)
    if l < 2:
        return s
    return recursive(s[l // 2:]) + recursive(s[:l // 2])

def iterative(s):
    r = list(s)
    i, j = 0, len(s) - 1
    while i < j:
        r[i], r[j] = r[j], r[i]
        i += 1
        j -= 1
    return "".join(r)




def pythonic(s):
    r = list(reversed(s))
    return "".join(r)


def ultra_pythonis(s):
    return s[::-1]


def reverse_vowel(s):
    vowels = "AEIOUaeiou"
    i, j = 0, len(s) - 1
    s = list(s)
    while i < j:
        while i < j and s[i] not in vowels:
            i += 1
        while i < j and s[j] not in vowels:
            j -= 1
        s[i], s[j] = s[j], s[i]
        i, j = i + 1, j - 1
    return "".join(s)


def reverse(array, i, j):
    while i < j:
        array[i], array[j] = array[j], array[i]
        i += 1
        j -= 1


def reverse_words(string):
    arr = list(string)
    n = len(arr)
    reverse(arr, 0, n - 1)
    start = None
    for i in range(n):
        if arr[i] == " ":
            if start is not None:
                reverse(arr, start, i - 1)
                start = None
            elif i == n - 1:
                if start is not None:
                    reverse(arr, start, i)
            else:
                if start is None:
                    start = i
    return "".join(arr)


def roman_to_int(s):
    number = 0
    roman = {'M': 1000, 'D': 500, 'C': 100, 'L': 50, 'X': 10, 'V': 5, 'I': 1}
    for i in range(len(s) - 1):
        if roman[s[i]] < roman[s[i + 1]]:
            number -= roman[s[i]]
        else:
            number += roman[s[i]]
    return number + roman[s[-1]]


import collections


def word_squares(words):
    n = len(words[0])
    fulls = collections.defaultdict(list)
    for word in words:
        for i in range(n):
            fulls[word[:i]].append(word)

    def build(square):
        if len(square) == n:
            squares.append(square)
            return
        prefix = ""
        for k in range(len(square)):
            prefix += square[k][len(square)]
        for word in fulls[prefix]:
            build(square + [word])
        squares = []
        for word in words:
            build([word])
        return square


def is_palindromic(s):
    return all(s[i] == s[~i] for i in range(len(s) // 2))

#################################################################################################################
#EPI
import functools
import string
import sys
import random


def int_to_string(x):
    is_negative = False
    if x < 0:
        x, is_negative = -x, True
    s = []
    while True:
        s.append(chr(ord('0') + x % 10))
        x //= 10
        if x == 0:
            break
    return ('-' if is_negative else '') + ''.join(reversed(s))


def string_to_int(s):
    return functools.reduce(lambda running_sum, c: running_sum * 10 + string.digits.index(c), s[s[0] == '-':], 0) * (
        -1 if s[0] == '-' else 1)


def convert_base(num_as_string, b1, b2):
    def construct_from_base(num_as_int, base):
        return ('' if num_as_int == 0 else construct_from_base(num_as_int // base, base) + string.hexdigits[
            num_as_int % base].upper())

    is_negative = num_as_string[0] == '-'
    num_as_int = functools.reduce(
        lambda x, c: x * b1 + string.hexdigits.index(c.lower()),
        num_as_string[is_negative], 0
    )
    return ('-' if is_negative else '') + ('0' if num_as_int == 0 else construct_from_base(num_as_int, b2))


def ss_decode_col_id(col):
    return reduce(lambda result, c: result * 26 + ord(c) - ord('A') + 1, col, 0)


def replace_and_remove(size, s):
    write_idx, a_count = 0, 0
    for i in range(size):
        if s[i] != 'b':
            s[write_idx] = s[i]
            write_idx += 1
        if s[i] == 'a':
            a_count += 1
    cur_idx = write_idx - 1
    write_idx += a_count - 1
    final_size = write_idx + 1
    while cur_idx >= 0:
        if s[cur_idx] == 'a':
            s[write_idx - 1:write_idx + 1] = 'dd'
            write_idx -= 2
        else:
            s[write_idx] = s[cur_idx]
            write_idx -= 1
        cur_idx -= 1
    return final_size


def reverse_words(s):
    s.reverse()

    def reverse_range(s, start, end):
        while start < end:
            s[start], s[end] = s[end], s[start]
            start, end = start + 1, end - 1

    start = 0
    while True:
        end = s.find(b' ', start)
        if end < 0:
            break
        reverse_range(s, start, end - 1)
        start = end + 1
    reverse_range(s, start, len(s) - 1)


MAPPING = ('0', '1', 'ABC', 'DEF', 'GHI', 'JKL', 'MNO', 'PQRS', 'TUV', 'WXYZ')


def phone_mnemonic(phone_number):
    def phone_mnemonic_helpder(digit):
        if digit == len(phone_number):
            mnemonics.append(''.join(partial_menmonic))
        else:
            for c in MAPPING[int(phone_number[digit])]:
                partial_menmonic[digit] = c
                phone_mnemonic_helpder(digit + 1)

    mnemonics, partial_menmonic = [], [0] * len(phone_number)
    phone_mnemonic_helpder(0)
    return mnemonics


def look_and_say(n):
    def next_number(s):
        result,i=[],0
        while i<len(s):
            count=1
            while i+1 <len(s) and s[i]==s[i+1]:
                i+=1
                count+=1
            result.append(str(count)+s[i])
            i+=1
        return ''.join(result)
    s='1'
    for _ in range(1,n):
        s=next_number(s)
    return s

def roman_to_integer(s):
    T = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    return functools.reduce(lambda  val,i: val+(-T[s[i]]) if T[s[i]]<T[s[i+1]] else T[s[i]]), reversed(range(len(s)-1), T[s[-1]])

def get_valid_ip_address(s):
    def is_valid_part(s):
        return len(s)==1 or (s[0]!='0' and int(s)<=255)
    result, parts=[],[None]*4
    for i in range(1,min(4,len(s))):
        parts[0] =s[:i]
        if is_valid_part(parts[0]):
            for j in range(1,min(len(s)-i,4)):
                parts[1]=s[i:i+1]
                if is_valid_part(parts[i]):
                    for k in range(1,min(len(s)-i-j,4)):
                        parts[2],parts[3]=s[i+j:i+j+k],s[i+j+k:]
                        if is_valid_part(parts[2]) and is_valid_part(parts[3]):
                            result.append('.'.join(parts))
    return result

def snake_string(s):
    result=[]
    for i in range(1,len(s),4):
        result.append(s[i])
    for i in range(0,len(s),2):
        result.append(s[i])
    for i in range(3,len(s),4):
        result.append(s[i])
    return ''.join(result)


def decoding(s):
    count, result=0,[]
    for c in s:
        if c.isdigit():
            count=count*10+int(c)
        else:
            result.append(c*count)
            count=0
    return ''.join(result)

def encoding(s):
    result,count=[],1
    for i in range(1,len(s)+1):
        if i==len(s) or s[i]!=s[i-1]:
            result.append(str(count)+s[i-1])
            count=1
        else:
            count+=1
    return ''.join(result)


def rabin_karp(t, s):
    if len(s) > len(t):
        return -1  # s is not a substring of t.

    BASE = 26
    # Hash codes for the substring of t and s.
    t_hash = functools.reduce(lambda h, c: h * BASE + ord(c), t[:len(s)], 0)
    s_hash = functools.reduce(lambda h, c: h * BASE + ord(c), s, 0)
    power_s = BASE ** max(len(s) - 1, 0)  # The modulo result of BASE^|s-1|.

    for i in range(len(s), len(t)):
        # Checks the two substrings are actually equal or not, to protect
        # against hash collision.
        if t_hash == s_hash and t[i - len(s):i] == s:
            return i - len(s)  # Found a match.

        # Uses rolling hash to compute the hash code.
        t_hash -= ord(t[i - len(s)]) * power_s
        t_hash = t_hash * BASE + ord(t[i])

    # Tries to match s and t[-len(s):].
    if t_hash == s_hash and t[-len(s):] == s:
        return len(t) - len(s)
    return -1  # s is not a substring of t.