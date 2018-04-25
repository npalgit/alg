class TreeNode:
    def __init__(self,val=0):
        self.val=val
        self.left=None
        self.right=None

def isSameTree(p,q): #simply recursion
    if not p and not q:
        return True
    if p and q and p.val ==q.val:
        return isSameTree(p.left, q.left) and isSameTree(p.right, q.right)
    return False

def binaryTreePaths(root):
    res=[]
    if not root:
        return res
    DFS(res,root,str(root.val))
    return res

def DFS(res,root,cur):
    if not root.left and not root.right:
        res.append(cur)
    if root.left:
        DFS(res,root.left, cur+"->"+str(root.left.val))
    if root.right:
        DFS(res,root.right, cur+"->"+str(root.right.val))

def bintree2list(root):
    if not root:
        return root
    root=bintree2list_util(root)
    while root.left:
        root=root.left
    return root

def bintree2list_util(root):
    if not root:
        return root
    if root.left:
        left=bintree2list_util(root.left)
        while left.right:
            left=left.right
        left.right=root
        root.left =left
    if root.right:
        right =bintree2list_util(root.right)
        while right.left:
            right =right.left
        right.left=root
        root.right =right
    return root

def print_tree(root):
    while root:
        print(root.val)
        root=root.right

Node = TreeNode

class DeepestLeft:
    def __init__(self):
        self.depth = 0
        self.Node = None
def find_deepest_left(root,is_left,depth,res):
    if not root:
        return
    if is_left and depth>res.depth:
        res.depth=depth
        res.Node=root
    find_deepest_left(root.left, True, depth+1, res)
    find_deepest_left(root.right, False,depth+1,res)

def reverse(root):
    if not root:
        return
    root.left, root.right=root.right, root.left
    if root.left:
        reverse(root.left)
    if root.right:
        reverse(root.right)

def is_balance(root):
    return -1 !=get_depth(root)

def get_depth(root):
    if not root:
        return 0
    left=get_depth(root.left)
    right=get_depth(root.right)
    if abs(left-right)>1:
        return -1
    return 1+ max(left,right)
def is_balanced2(root):
    left=max_height(root.left)
    right=max_height(root.right)
    return abs(left-right)<=1 and isinstance(root.left) and isinstance(root.right)

def max_height(root):
    if not root:
        return 0
    return max(max_height(root.left), max_height(root.right))+1

import collections
def is_subtree(big,small):
    flag =False
    queue =collections.deque()
    queue.append(big)
    while queue:
        node=queue.popleft()
        if node.val==small.val:
            flag=comp(node,small)
            break
        else:
            queue.append(node.left)
            queue.append(node.right)
    return flag

def comp(p,q):
    if not p and not q:
        return True
    if p and q:
        return p.val ==q.val  and comp(p.left, q.left) and comp(p.right, q.right)
    return False

def is_symmetric(root):
    if not root:
        return True
    return helper(root.left,root.right)
def helper(p,q):
    if not p and not q:
        return True
    if not p or not q or q.val!=p.val:
        return False
    return helper(p.left,q.right) and helper(p.right,q.left)

def is_symmentric_iterative(root):
    if not root:
        return True
    stack=[[root.left, root.right]]
    while stack:
        left,right=stack.pop()
        if not left and not right:
            continue
        if not left or not right:
            return False
        if left.val==right.val:
            stack.append([left.left,right.right])
            stack.append([left.right,right.right])
        else:
            return False
    return True

maxlen=0
def longestConsecutive(root):
    if not root:
        return 0
    DFS2(root,0,root.val)
    return maxlen

def DFS2(root, cur,target):
    if not root: return
    if root.val ==target:
        cur+=1
    else:
        cur=1
    maxlen=max(cur, maxlen)
    DFS2(root.left, cur, root.val+1)
    DFS2(root.right,cur, root.val+1)


def LCA(root,p,q):
    if not root or root is p or root is q:
        return root
    left=LCA(root.left, p,q)
    right=LCA(root.righ,p,q)
    if left and right:
        return root
    return left if left else right

def max_height(root):
    if not root:
        return 0
    height=0
    queue=[root]
    while queue:
        height+=1
        level=[]
        while queue:
            node=queue.pop(0)
            if node.left:
                level.append(node.left)
            if node.right:
                level.append(node.right)
        queue=level
    return height

maximum=float("-inf")
def max_path_sum(root):
    helper(root)
    return maximum
def helper(root):
    if not root:
        return 0
    left =helper(root.left)
    right=helper(root.right)
    maximum =max(maximum,left+right+root.val)
    return root.val + max(left,right)

def minDepth(root):
    if not root:
        return 0
    if not root.left or not root.right:
        return max(minDepth(root.left) , minDepth(root.right))+1
    return min(minDepth(root.left),minDepth(root.right))+1


def min_height(root):
    if not root:
        return 0
    height =0
    level =[root]
    while level:
        height+=1
        new_level=[]
        for node in level:
            if not node.left and not node.right:
                return height
            if node.left:
                new_level.append(node.left)
            if node.right:
                new_level.append(node.right)
        level=new_level
    return height


def has_path_sum(root,sum):
    if not root:
        return False
    if not root.left and not root.right and root.val ==sum:
        return True
    sum-=root.val
    return has_path_sum(root.left,sum) or has_path_sum(root.right, sum)

def has_path_sum2(root,sum):
    if not root:
        return False
    stack=[(root,root.val)]
    while stack:
        node,val=stack.pop()
        if not node.left and not node.right:
            if val==sum:
                return True
        if node.left:
            stack.append((node.left,val+node.left.val))
        if node.right:
            stack.append((node.right,val+node.right.val))
    return False

def has_path_sum3(root,sum):
    if not root:
        return False
    queue =[(root,sum-root.val)]
    while queue:
        node,val=queue.pop(0)
        if not node.left and not node.right:
            if val==0:
                return True
        if node.left:
            queue.append((node.left, val-node.left.val))
        if node.right:
            queue.append((node.right, val-node.right.val))
    return False


def path_sum(root,sum):
    if not root:
        return []
    res=[]
    DFS3(root,sum,[],res)
    return res

def DFS3(root,sum,ls,res):
    if not root.left and not root.right and root.val ==sum:
        ls.append(root.val)
        res.append(ls)
    if root.left:
        DFS3(root.left, sum-root.val, ls+[root.val],res)
    if root.right:
        DFS3(root.right, sum-root.val,ls+[root.val],res)

def path_sum2(root,s):
    if not root:
        return []
    res=[]
    stack=[(root,[root.val])]
    while stack:
        node,ls=stack.pop()
        if not node.left and not node.right and sum(ls)==s:
            res.append(ls)
        if node.left:
            stack.append((node.left,ls+[node.left.val]))
        if node.right:
            stack.append((node.right, ls+[node.right.val]))
    return res

def path_sum3(root,sum):
    if not root:
        return []
    res=[]
    queue=[(root,root.val,[root.val])]
    while queue:
        node,val,ls=queue.pop(0)
        if not node.left and not node.right and val==sum:
            res.append(ls)
        if node.left:
            queue.append((node.left,val+node.left.val, ls+[node.left.val]))
        if node.right:
            queue.append((node.right, val+node.right.val,ls+[node.right.val]))
    return res


def treePrint(tree):
    for key in tree:
        print(key)
        treeElem =tree[key]
        for subElem in treeElem:
            print("->"+subElem)
            if type(subElem)!=str:
                print("\n")
        print("")

def array2bst(nums):
    if not nums:
        return None
    mid=len(nums)//2
    node=Node(nums[mid])
    node.left =array2bst(nums[:mid])
    node.right=array2bst(nums[mid+1:])
    return node

def closest_value(root,target):
    a = root.val
    kid=root.left if target< a else root.right
    if not kid:
        return a
    b =closest_value(kid,target)
    return min((a,b) , key= lambda  x: abs(target-x))

class BSTIterator:
    def __int__(self,root):
        self.stack=[]
        while root:
            self.stack.append(root)
            root=root.left
    def has_next(self):
        return bool(self.stack)
    def next(self):
        node=self.stack.pop()
        tmp=node
        if tmp.right:
            tmp=tmp.right
            while tmp:
                self.stack.append(tmp)
                tmp=tmp.left
        return node.val

def deleteNode(root,key):
    if not root: return None
    if root.val ==key:
        if root.left:
            left_right_most = root.left
            while left_right_most.right:
                left_right_most=left_right_most.right
            left_right_most.right=root.right
            return root.left
        else:
            return root.right
    elif root.val >key:
        root.left =deleteNode(root.left,key)
    else:
        root.right=deleteNode(root.right,key)
    return root

def is_BST(root):
    if not root:
        return True
    stack=[]
    pre=None
    while root and stack:
        while root:
            stack.append(root)
            root=root.left
        root=stack.pop()
        if pre and root.val<=pre.val:
            return False
        pre=root
        root =root.right
    return True

def kth_smallest(root,k):
    stack=[]
    while root or stack:
        while root:
            stack.append(root)
            root = root.left
        root = stack.pop()
        k-=1
        if k==0:
            break
        root = root.right
    return root.val

def kth_smallest2(root,k):
    count =[]
    helper2(root,count)
    return count[k-1]

def helper2(node,count):
    if not node:
        return
    helper2(node.left, count)
    count.append(node.val)
    helper2(node.right,count)

def lowest_common_ancestor(root,p,q):
    while root:
        if p.val > root.val < q.val:
            root=root.right
        elif p.val <root.val > q.val:
            root =root.left
        else:
            return root

def predecessor(root,node):
    pred=None
    while root:
        if node.val >root.val:
            pred=root
            root=root.right
        else:
            root=root.left
    return pred

def serialize(root):
    def build_string(node):
        if node:
            vals.append(str(node.val))
            build_string(node.left)
            build_string(node.right)
        else:
            vals.append("#")
    vals=[]
    build_string(root)
    return " ".join(vals)

def deserialize(data):
    def build_tree():
        val=next(vals)
        if val=="#":
            return None
        node=TreeNode(int(val))
        node.left=build_tree()
        node.right=build_tree()
        return node
    vals=iter(data.split())
    return build_tree()

def successor(root,node):
    succ =None
    while root:
        if node.val<root.val:
            succ=root
            root=root.left
        else:
            root=root.right
    return succ

def num_trees(n):
    dp=[0]*(n+1)
    dp[0]=1
    dp[1]=1
    for i in range(2,n+1):
        for j in range(i+1):
            dp[i]+=dp[i-j]*dp[j-1]
    return dp[-1]

class segment_tree:
    def __init__(self,arr,function):
        self.segment=[0 for x in range(3*len(arr)+3)]
        self.arr= arr
        self.fn=function
        self.maketree(0,0,len(arr)-1)
    def maketree(self,i,l,r):
        if l==r:
            self.segment[i]=self.arr[l]
        elif l<r:
            self.maketree(2*i+1, l, int(l+r)/2)
            self.maketree(2*i,int(l+r)/2+1, r)
            self.segment[i]= self.fn(self.segment[2*i+1], self.segment[2*i+2])
    def __query(self,i,L,R,l,r):
        if l>R or r<L or L>R or l>r:
            return None
        if L>=l and R<=r:
            return self.segment[i]
        val1=self.__query(2*i+1, L, int(L+R)/2, l, r)
        val2=self.__query(2*i+2,int(l+R)/2, R,l,r)
        print(L,R,"returned",val1,val2)
        if val1!=None:
            if val2!=None:
                return self.fn(val1,val2)
            return val1
        return val2

    def query(self,L,R):
        return self.__query(0,0,len(self.arr)-1,L,R)

def inorder(root):
    res=[]
    if not root:
        return res
    stack =[]
    while root or stack:
        while root:
            stack.append(root)
            root=root.left
        root=stack.pop()
        res.append(root.val)
        root=root.right
    return res

def level_order(root):
    ans=[]
    if not root:
        return ans
    level=[root]
    while level:
        current=[]
        new_level=[]
        for node in level:
            current.append(node.val)
            if node.left:
                new_level.append(node.left)
            if node.right:
                new_level.append(node.right)
        level=new_level
        ans.append(current)
    return ans

def zigzag_level(root):
    res=[]
    if not root:
        return res
    level =[root]
    flag=1
    while level:
        current=[]
        new_level=[]
        for node in level:
            current.append(node.val)
            if node.left:
                new_level.append(node.left)
            if node.right:
                new_level.append(node.right)
        level=new_level
        res.append(current[::flag])
        flag*=-1
    return res

class TrieNode:
    def __int__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_word=False
class Trie:
    def __int__(self):
        self.root=TrieNode()
    def insert(self, word):
        current =self.root
        for letter in word:
            current =current.children[letter]
        current.is_word=True
    def search(self,word):
        current =self.root
        for letter in word:
            current = current.children.get(letter)
            if current is None:
                return False
        return current.is_word

    def startsWith(self,prefix):
        current =self.root
        for letter in prefix:
            current=current.children.get(letter)
            if current is None:
                return False
        return True


class TrieNode2(object):
    def __int__(self,letter, is_terminal=False):
        self.children=dict()
        self.letter=letter
        self.is_terminal=is_terminal
class WordDictionary(object):
    def __int__(self):
        self.root=TrieNode2("")

    def addWord(self,word):
        cur =self.root
        for letter in word:
            if letter not in cur.children:
                cur.children[letter]=TrieNode2(letter)
            cur=cur.children[letter]
        cur.is_terminal=True
    def search(self,word,node=None):
        cur=node
        if not cur:
            cur=self.root
        for i,letter in enumerate(word):
            if letter ==".":
                if i==len(word)-1:
                    for child in cur.children.values():
                        if child.is_terminal:
                            return True
                for child in cur.children.values():
                    if self.search(word[i+1:],child)==True:
                        return True
                return False
            if letter not in cur.children:
                return False
            cur=cur.children[letter]
        return cur.is_terminal
class WordDictionary2(object):
    def __init__(self):
        self.word_dict=collections.defaultdict(list)
    def addWord(self,word):
        if word:
            self.word_dict[len(word)].append(word)
    def search(self,word):
        if not word:
            return False
        if '.' not in word:
            return word in self.word_dict[len(word)]
        for v in self.word_dict[len(word)]:
            for i,ch in enumerate(word):
                if ch!=v[i] and ch!='.':
                    break
            else:
                return True
        return False
##############################################################################
# @include
class BinaryTreeNode:

    def __init__(self, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right
# @exclude

    def __repr__(self):
        return '%s <- %s -> %s' % (self.left and self.left.data, self.data,
                                   self.right and self.right.data)

def tree_traversal(root):
    if root:
        print("pre"+root.data)
        tree_traversal(root.left)
        print("mid"+root.data)
        tree_traversal(root.right)
        print("post"+root.data)

def is_balanced_binary_tree(tree):
    BalancedStatusWithHeight = collections.namedtuple("BalancedStatusWithHeight",('balanced','height'))
    def check_balanced(tree):
        if not tree:
            return BalancedStatusWithHeight(True,-1)
        left_result=check_balanced(tree.left)
        if not left_result.balanced:
            return BalancedStatusWithHeight(False,0)
        right_result=check_balanced(tree.right)
        if not right_result.balanced:
            return BalancedStatusWithHeight(False,0)
        is_balanced=abs(left_result.height-right_result.height)<=1
        height=max(left_result.height,right_result.height)+1
        return BalancedStatusWithHeight(is_balanced,height)
    return check_balanced(tree).balanced

def is_symmetric(tree):
    def check_symmetric(subtree_0, subtree_1):
        if not subtree_0 and not subtree_1:
            return True
        elif subtree_0 and subtree_1:
            return (subtree_0.data == subtree_1.data and check_symmetric(subtree_0.left, subtree_1.right) and check_symmetric(subtree_0.right,subtree_1.left))
        return False
    return not tree or check_symmetric(tree.left, tree.right)

def lca(tree,node0,node1):
    Status =collections.namedtuple("Status",('num_target_nodes','ancestor'))
    def lca_helper(tree,node0,node1):
        if not tree:
            return Status(0,None)
        left_result=lca_helper(tree.left,node0,node1)
        if left_result.num_target_nodes==2:
            return left_result
        right_result=lca_helper(tree.right,node0,node1)
        if right_result.num_target_nodes==2:
            return right_result
        num_target_nodes=(left_result.num_target_nodes+right_result.num_target_nodes+int( tree in (node0,node1)))
        return Status(num_target_nodes,tree if num_target_nodes==2 else None)
    return lca_helper(tree,node0,node1).ancestor

def lca(node_0,node_1):
    def get_depth(node):
        depth=0
        while node:
            depth+=1
            node=node.parent
        return depth
    depth_0,depth_1 =get_depth(node_0), get_depth(node_1)
    if depth_1>depth_0:
        node_0,node_1 =node_1, node_0
    depth_diff =abs(depth_0-depth_1)
    while depth_diff:
        node_0= node_0.parent
        depth_diff-=1
    while node_0 is not node_1:
        node_0,node_1=node_0.parent,node_1.parent
    return node_0

def sum_root_to_leaf(tree, partial_path_sum=0):
    if not tree:
        return 0
    partial_path_sum=partial_path_sum*2 + tree.data
    if not tree.left and not tree.right:
        return partial_path_sum
    return sum_root_to_leaf(tree.left, partial_path_sum)+ sum_root_to_leaf(tree.right,partial_path_sum)

def bst_in_sorted_order(tree):
    s,result=[],[]
    while s or tree:
        if tree:
            s.append(tree)
            tree =tree.left
        else:
            tree=s.pop()
            result.append(tree.data)
            tree=tree.right
    return result

def preorder_traversal(tree):
    path,result=[tree],[]
    while path:
         curr =path.pop()
         if curr:
             result.append(curr.data)
             path+=[curr.right, curr.left]
    return result

def find_kth_node_binary_tree(tree,k):
    while tree:
        left_size=tree.left.size if tree.left else 0
        if left_size+1<k:
            k-=left_size+1
            tree=tree.right
        elif left_size==k-1:
            return tree
        else:
            tree=tree.left
    return None

def find_successor(node):
    if node.right:
        node=node.right
        while node.left:
            node=node.left
        return node
    while node.parent and node.parent.right is node:
        node=node.parent
    return node.parent

def inorder_traversal(tree):
    prev,result=None,[]
    while tree:
        if prev is tree.parent:
            if tree.left:
                next=tree.left
            else:
                result.append(tree.data)
                next=tree.right or tree.parent

        elif tree.left is prev:
            result.append(tree.data)
            next=tree.right or tree.parent

        else:
            next=tree.parent
        prev,tree=tree,next
    return result

def binary_tree_from_preorder_inorder(preorder,inorder):
    node_to_inorder_idx={data:i for i, data in enumerate(inorder)}
    def binary_tree_from_preorder_inorder_helper(preorder_start, preorder_end, inorder_start, inorder_end):
        if preorder_end<=preorder_start or inorder_end<=inorder_start:
            return None
        root_inorder_idx = node_to_inorder_idx[preorder[preorder_start]]
        left_subtree_size = root_inorder_idx-inorder_start
        return BinaryTreeNode(
            preorder[preorder_start],
            binary_tree_from_preorder_inorder_helper(preorder_start+1,preorder_start+1+left_subtree_size,inorder_start,root_inorder_idx),
            binary_tree_from_preorder_inorder_helper(preorder_start+1+left_subtree_size, preorder_end, root_inorder_idx+1, inorder_end)
        )
    return binary_tree_from_preorder_inorder_helper(0,len(preorder),0,len(inorder))

def reconstruct_preorder(preorder):
    def reconstruct_preorder_helper(preorder_iter):
        subtree_key=next(preorder_iter)
        if subtree_key is None:
            return None
        left_subtree=reconstruct_preorder_helper(preorder_iter)
        right_subtree=reconstruct_preorder_helper(preorder_iter)
        return BinaryTreeNode(subtree_key,left_subtree,right_subtree)
    return reconstruct_preorder_helper(iter[preorder])

def create_list_of_leaves(tree):
    if not tree:
        return []
    if not tree.left and not tree.right:
        return [tree]
    return create_list_of_leaves(tree.left) + create_list_of_leaves(tree.right)

def exterior_binary_tree(tree):
    def is_leaf(node):
        return not node.left and not node.right
    def left_boundary_and_leaves(subtree, is_boundary):
        if not subtree:
            return []
        return (([subtree] if is_boundary or is_leaf(subtree) else [])+
                left_boundary_and_leaves(subtree.left, is_boundary)+
                left_boundary_and_leaves(subtree.right ,is_boundary and not subtree.left))
    def right_boundary_and_leaves(subtree, is_boundary):
        if not subtree:
            return []
        return (right_boundary_and_leaves(subtree.left, is_boundary and not subtree.right) +
                right_boundary_and_leaves(subtree.right, is_boundary) +
                ([subtree] if is_boundary or is_leaf(subtree) else []))
    return ([tree]+left_boundary_and_leaves(tree.left, True) + right_boundary_and_leaves(tree.right ,True) if tree else [])

def construct_right_siblint(tree):
    def populate_children_next_field(start_node):
        while start_node and start_node.left:
            start_node.left.next =start_node.right
            start_node.right.next=start_node.next and start_node.next.left
            start_node=start_node.next
    while tree and tree.left:
        populate_children_next_field(tree)
        tree=tree.left
class BinaryTreeNode:
    def __init__(self):
        self.left=self.right=self.parent=None
        self._locked, self._num_locked_descendants=False,0
    def is_locked(self):
        return self._locked
    def lock(self):
        if self._num_locked_descendants >0 or self._locked:
            return False
        it=self.parent
        while it:
            if it._locked:
                return False
            it=it.parent
        self._locked=True
        it=self.parent
        while it:
            it._num_locked_descendants+=1
            it=it.parent
        return True
    def unlock(self):
        if self._locked:
            self._locked=False
            it=self.parent
            while it:
                it._num_locked_descendants-=1
                it=it.parent

# @include
class BSTNode:

    def __init__(self, data=None, left=None, right=None):
        self.data, self.left, self.right = data, left, right
# @exclude

def search_bst(tree, key):
    return (tree if not tree or tree.data == key else search_bst(tree.left, key)
            if key < tree.data else search_bst(tree.right, key))

# @include
def is_binary_tree_bst(tree, low_range=float('-inf'), high_range=float('inf')):
    if not tree:
        return True
    elif not low_range <= tree.data <= high_range:
        return False
    return (is_binary_tree_bst(tree.left, low_range, tree.data) and
            is_binary_tree_bst(tree.right, tree.data, high_range))
# @exclude


def is_binary_tree_bst_alternative(tree):
    def inorder_traversal(tree):
        if not tree:
            return True
        elif not inorder_traversal(tree.left):
            return False
        elif prev[0] and prev[0].data > tree.data:
            return False
        prev[0] = tree
        return inorder_traversal(tree.right)
    prev = [None]
    return inorder_traversal(tree)

def is_binary_tree_bst(tree):
    QueueEntry = collections.namedtuple('QueueEntry',
                                        ('node', 'lower', 'upper'))
    bfs_queue = collections.deque(
        [QueueEntry(tree, float('-inf'), float('inf'))])

    while bfs_queue:
        front = bfs_queue.popleft()
        if front.node:
            if not front.lower <= front.node.data <= front.upper:
                return False
            bfs_queue += [
                QueueEntry(front.node.left, front.lower, front.node.data),
                QueueEntry(front.node.right, front.node.data, front.upper)
            ]
    return True

def find_first_greater_than_k(tree, k):
    subtree, first_so_far = tree, None
    while subtree:
        if subtree.data > k:
            first_so_far, subtree = subtree, subtree.left
        else:  # Root and all keys in left subtree are <= k, so skip them.
            subtree = subtree.right
    return first_so_far

def find_k_largest_in_bst(tree, k):
    def find_k_largest_in_bst_helper(tree):
        # Perform reverse inorder traversal.
        if tree and len(k_largest_elements) < k:
            find_k_largest_in_bst_helper(tree.right)
            if len(k_largest_elements) < k:
                k_largest_elements.append(tree.data)
                find_k_largest_in_bst_helper(tree.left)

    k_largest_elements = []
    find_k_largest_in_bst_helper(tree)
    return k_largest_elements

def find_LCA(tree, s, b):
    while tree.data < s.data or tree.data > b.data:
        # Keep searching since tree is outside of [s, b].
        while tree.data < s.data:
            tree = tree.right  # LCA must be in tree's right child.
        while tree.data > b.data:
            tree = tree.left  # LCA must be in tree's left child.
    # Now, s.data <= tree.data && tree.data <= b.data.
    return tree

def rebuild_bst_from_preorder(preorder_sequence):
    def rebuild_bst_from_preorder_on_value_range(lower_bound, upper_bound):
        if root_idx[0] == len(preorder_sequence):
            return None

        root = preorder_sequence[root_idx[0]]
        if not lower_bound <= root <= upper_bound:
            return None
        root_idx[0] += 1
        # Note that rebuild_bst_from_preorder_on_value_range updates root_idx[0].
        # So the order of following two calls are critical.
        left_subtree = rebuild_bst_from_preorder_on_value_range(lower_bound,
                                                                root)
        right_subtree = rebuild_bst_from_preorder_on_value_range(root,
                                                                 upper_bound)
        return BinaryTreeNode(root, left_subtree, right_subtree)

    root_idx = [0]  # Tracks current subtree.
    return rebuild_bst_from_preorder_on_value_range(float('-inf'), float('inf'))

def find_closest_elements_in_sorted_arrays(sorted_arrays):
    min_distance_so_far = float('inf')
    # Stores array iterators in each entry.
    iters = bintrees.RBTree()
    for idx, sorted_array in enumerate(sorted_arrays):
        it = iter(sorted_array)
        first_min = next(it, None)
        if first_min is not None:
            iters.insert((first_min, idx), it)

    while True:
        min_value, min_idx = iters.min_key()
        max_value = iters.max_key()[0]
        min_distance_so_far = min(max_value - min_value, min_distance_so_far)
        it = iters.pop_min()[1]
        next_min = next(it, None)
        # Return if some array has no remaining elements.
        if next_min is None:
            return min_distance_so_far
        iters.insert((next_min, min_idx), it)
class ABSqrt2:

    def __init__(self, a, b):
        self.a, self.b = a, b
        self.val = a + b * math.sqrt(2)

    def __lt__(self, other):
        return self.val < other.val

    def __eq__(self, other):
        return self.val == other.val
# @exclude

    def __hash__(self):
        return self.a ^ self.b

    def __repr__(self):
        return r'%d + %d \/2' % (self.a, self.b)


def generate_first_k_a_b_sqrt2(k):
    # Will store the first k numbers of the form a + b sqrt(2).
    result = [ABSqrt2(0, 0)]
    i = j = 0
    for _ in range(1, k):
        result_i_plus_1 = ABSqrt2(result[i].a + 1, result[i].b)
        result_j_plus_sqrt2 = ABSqrt2(result[j].a, result[j].b + 1)
        result.append(min(result_i_plus_1, result_j_plus_sqrt2))
        if result_i_plus_1.val == result[-1].val:
            i += 1
        if result_j_plus_sqrt2.val == result[-1].val:
            j += 1
    return result


def build_min_height_bst_from_sorted_array(A):
    def build_min_height_bst_from_sorted_subarray(start, end):
        if start >= end:
            return None
        mid = (start + end) // 2
        return BinaryTreeNode(
            A[mid],
            build_min_height_bst_from_sorted_subarray(start, mid),
            build_min_height_bst_from_sorted_subarray(mid + 1, end))

    return build_min_height_bst_from_sorted_subarray(0, len(A))

class BinarySearchTree:

    def __init__(self):
        self._root = None

    # @exclude
    def empty(self):
        return self._root is None

    # @include
    def insert(self, key):
        if self.empty():
            self._root = BinaryTreeNode(key)
        else:
            parent = None
            curr = self._root
            while curr:
                parent = curr
                if key == curr.data:
                    # key already present, no duplicates to be added.
                    return False
                elif key < curr.data:
                    curr = curr.left
                else:  # key > curr.data.
                    curr = curr.right

            # Inserts key according to key and parent.
            if key < parent.data:
                parent.left = BinaryTreeNode(key)
            else:
                parent.right = BinaryTreeNode(key)
        return True

    def delete(self, key):
        # Find the node with key.
        curr = self._root
        parent = None
        while curr and curr.data != key:
            parent = curr
            curr = curr.left if key < curr.data else curr.right

        if not curr:
            # There's no node with key in this tree.
            return False

        key_node = curr
        if key_node.right:
            # Finds the minimum of the right subtree.
            r_key_node = key_node.right
            r_parent = key_node
            while r_key_node.left:
                r_parent = r_key_node
                r_key_node = r_key_node.left
            key_node.data = r_key_node.data
            # Moves links to erase the node.
            if r_parent.left == r_key_node:
                r_parent.left = r_key_node.right
            else:  # r_parent.right == r_key_node.
                r_parent.right = r_key_node.right
        else:
            # Updates _root link if needed.
            if self._root == key_node:
                self._root = key_node.left
            else:
                if parent.left == key_node:
                    parent.left == key_node.left
                else:  # parent.right == key_node.
                    parent.right = key_node.left
        return True
    # @exclude

    def get_root_val(self):
        return self._root.data


def pair_includes_ancestor_and_descendant_of_m(possible_anc_or_desc_0,
                                               possible_anc_or_desc_1, middle):
    search_0, search_1 = possible_anc_or_desc_0, possible_anc_or_desc_1

    # Perform interleaved searching from possible_anc_or_desc_0 and
    # possible_anc_or_desc_1 for middle.
    while (search_0 is not possible_anc_or_desc_1 and search_0 is not middle and
           search_1 is not possible_anc_or_desc_0 and search_1 is not middle and
           (search_0 or search_1)):
        if search_0:
            search_0 = (search_0.left if search_0.data >
                        middle.data else search_0.right)
        if search_1:
            search_1 = (search_1.left if search_1.data >
                        middle.data else search_1.right)

    # If both searches were unsuccessful, or we got from
    # possible_anc_or_desc_0 to possible_anc_or_desc_1 without seeing middle,
    # or from possible_anc_or_desc_1 to possible_anc_or_desc_0 without seeing
    # middle, middle cannot lie between possible_anc_or_desc_0 and
    # possible_anc_or_desc_1.
    if ((search_0 is not middle and search_1 is not middle) or
            search_0 is possible_anc_or_desc_1 or
            search_1 is possible_anc_or_desc_0):
        return False

    def search_target(source, target):
        while source and source is not target:
            source = source.left if source.data > target.data else source.right
        return source is target

    # If we get here, we already know one of possible_anc_or_desc_0 or
    # possible_anc_or_desc_1 has a path to middle. Check if middle has a path
    # to possible_anc_or_desc_1 or to possible_anc_or_desc_0.
    return search_target(
        middle,
        possible_anc_or_desc_1 if search_0 is middle else possible_anc_or_desc_0)



Interval = collections.namedtuple('Interval', ('left', 'right'))


def range_lookup_in_bst(tree, interval):
    def range_lookup_in_bst_helper(tree):
        if tree is None:
            return

        if interval.left <= tree.data <= interval.right:
            # tree.data lies in the interval.
            range_lookup_in_bst_helper(tree.left)
            result.append(tree.data)
            range_lookup_in_bst_helper(tree.right)
        elif interval.left > tree.data:
            range_lookup_in_bst_helper(tree.right)
        else:  # interval.right > tree.data
            range_lookup_in_bst_helper(tree.left)

    result = []
    range_lookup_in_bst_helper(tree)
    return result

class ClientsCreditsInfo:

    def __init__(self):
        self._offset = 0
        self._client_to_credit = {}
        self._credit_to_clients = bintrees.RBTree()

    def insert(self, client_id, c):
        self.remove(client_id)
        self._client_to_credit[client_id] = c - self._offset
        self._credit_to_clients.setdefault(c - self._offset,
                                           set()).add(client_id)

    def remove(self, client_id):
        credit = self._client_to_credit.get(client_id)
        if credit is not None:
            self._credit_to_clients[credit].remove(client_id)
            if not self._credit_to_clients[credit]:
                del self._credit_to_clients[credit]
            del self._client_to_credit[client_id]
            return True
        return False

    def lookup(self, client_id):
        credit = self._client_to_credit.get(client_id)
        return -1 if credit is None else credit + self._offset

    def add_all(self, C):
        self._offset += C

    def max(self):
        if not self._credit_to_clients:
            return ''
        clients = self._credit_to_clients.max_item()[1]
        return '' if not clients else next(iter(clients))