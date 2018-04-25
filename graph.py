from collections import deque
class UndirectedGraphNode:
    def __init__(self, x):
        self.label = x
        self.neighbors = []
def cloneGraph1(self,node):
    if not node:
        return
    nodeCopy = UndirectedGraphNode(node.label)
    dic={node:nodeCopy}
    queue=deque([node])
    while queue:
        node=queue.popleft()
        for neighbor in node.neighbors:
            if neighbor not in dic:
                neighborCopy=UndirectedGraphNode(neighbor.label)
                dic[neighbor]=neighborCopy
                dic[node].neighbors.append(neighborCopy)
                queue.append(neighbor)
            else:
                dic[node].neighbors.append(dic[neighbor])
    return nodeCopy

def cloneGraph2(self,node):
    if not node:
        return
    nodeCopy=UndirectedGraphNode(node.label)
    dic={node:nodeCopy}
    stack=[node]
    while stack:
        node=stack.pop()
        for neighbor in node.neighbors:
            if neighbor not in dic:
                neighborCopy = UndirectedGraphNode(neighbor.label)
                dic[neighbor]=neighborCopy
                dic[node].neighbors.append(neighborCopy)
                stack.append(neighbor)
            else:
                dic[node].neighbors.append(dic[neighbor])
    return nodeCopy

def cloneGraph(self,node):
    if not node:
        return
    nodeCopy = UndirectedGraphNode(node.label)
    dic={node:nodeCopy}
    self.dfs(node,dic)
    return nodeCopy

def dfs(self,node,dic):
    for neighbor in node.neighbors:
        if neighbor not in dic:
            neighborCopy = UndirectedGraphNode(neighbor.label)
            dic[neighbor]= neighborCopy
            dic[node].neighbors.append(neighborCopy)
            self.dfs(neighbor,dic)
        else:
            dic[node].neighbors.append(dic[neighbor])

from enum import Enum
class TraversalState(Enum):
    WHITE=0
    GRAY=1
    BLACK=2

def is_in_cycle(graph, traversal_states, vertex):
    if traversal_states[vertex]==TraversalState.GRAY:
        return True
    traversal_states[vertex]=TraversalState.GRAY
    for neighbor in graph[vertex]:
        if is_in_cycle(graph, traversal_states,neighbor):
            return True
    traversal_states[vertex]=TraversalState.BLACK
    return False

def contains_cycle(graph):
    traversal_states ={vertex: TraversalState.WHITE for vertex in graph}
    for vertex, state in traversal_states.items():
        if state==TraversalState.WHITE and is_in_cycle(graph,traversal_states,vertex):
            return True
    return False

def find_all_cliques(edges):
    def expand_clique(candidates, nays):
        nonlocal compsub
        if not candidates and not nays:
            nonlocal  solutions
            solutions.append(compsub.copy())
        else:
            for selected in candidates.copy():
                candidates.remove(selected)
                candidates_temp = get_connected(selected,candidates)
                nays_temp=get_connected(selected,nays)
                compsub.append(selected)
                expand_clique(candidates_temp,nays_temp)
                nays.add(compsub.pop())
    def get_connected(vertex, old_set):
        new_set =set()
        for neighbor in edges[str(vertex)]:
            if neighbor in old_set:
                new_set.add(neighbor)
        return new_set

    compsub=[]
    solutions=[]
    possibles=set(edges.keys())
    expand_clique(possibles,set())
    return solutions

myGraph = {'A': ['B', 'C'],
         'B': ['C', 'D'],
         'C': ['D', 'F'],
         'D': ['C'],
         'E': ['F'],
         'F': ['C']}

def find_path(graph, start ,end, path=[]):
    path=path+[start]
    if start ==end:
        return path
    if not start in graph:
        return None
    for node in graph[start]:
        if node not in path:
            newpath=find_path(graph, node,end, path)
            return newpath

    return Node

def find_all_path(graph,start, end,path=[]):
    path=path+[start]
    print(path)
    if start==end:
        return [path]
    if not start in graph:
        return None
    paths=[]
    for node in graph[start]:
        if node not in path:
            newpaths=find_all_path(graph,node,end,path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths

def find_shortest_path(graph, start, end, path=[]):
    path=path+[start]
    if start == end:
        return path
    if start not in graph:
        return None
    shortest=None
    for node in graph[start]:
        if node not in path:
            newpath=find_shortest_path(graph,start, end, path)
            if newpath:
                if not shortest or len(newpath)<len(shortest):
                    shortest=newpath
    return shortest

def __choose_state(state_map):
    choice=random.random()
    probability_reached =0
    for state, probability in state_map.items():
        probability_reached+=probability
        if probability_reached>choice:
            return state

def next_state(chain, current_state):
    next_state_map=chain.get(current_state)
    next_state = __choose_state(next_state_map)
    return next_state

def iterating_markov_chain(chain,state):
    while True:
        state=next_state(chain,state)
        yield state


class Edge:
    def __init__(self,u,v,weight):
        self.u=u
        self.v=v
        self.weight=weight
class DisjointSet:
    def __init__(self,n):
        self.parent=[None]*n
        self.size=[1]*n
        for i in range(n):
            self.parent[i]=i

    def mergeSet(self,a,b):
        a=self.findSet(a)
        b=self.findSet(b)

        if self.size[a]<self.size[b]:
            self.parent[a]=b
            self.size[b]+=self.size[a]
        else:
            self.parent[b]=a
            self.size[a]+=self.size[b]
    def findSet(self,a):
        if self.parent[a]!=a:
            self.parent[a]=self.findSet(self.parent[a])

        return self.parent[a]

def kruskal(n,edges,ds):
    edges.sort(key=lambda  edge: edge.weight)
    mst=[]
    for edge in edges:
        set_u=ds.findSet(edge.u)
        set_v=ds.findSet(edge.v)
        if set_u != set_v:
            ds.mergeSet(set_u,set_v)
            mst.append(edge)
            if len(mst)==n-1:
                break
    return sum([edge.weight for edge in mst])

def dfs_transposed(v,graph,order,vis):
    vis[v]=True
    for u in graph[v]:
        if not vis[u]:
            dfs_transposed(u,graph,order,vis)
    order.append(v)

def dfs(v,current_comp, vertex_scc, graph, vis):
    vis[v]=True
    vertex_scc[v]=current_comp
    for u in graph[v]:
        if not vis[u]:
            dfs(u,current_comp,vertex_scc,graph,vis)

def add_edge(graph, vertex_from, vertex_to):
    if vertex_from not in graph:
        graph[vertex_from]=[]
    graph[vertex_from].append(vertex_to)
def scc(graph):
    order=[]
    vis={vertex:False for vertex in graph}
    graph_transposed ={vertex:[ ] for vertex in graph}
    for (v,neighbors) in graph.items():
        for u in neighbors:
            add_edge(graph_transposed,u,v)
    for v in graph:
        if not vis[v]:
            dfs_transposed(v,graph_transposed,order,vis)
    vis={vertex:False for vertex in graph}
    vertex_scc={}

    current_comp=0

    for v in reversed(order):
        if not vis[v]:
            dfs(v, current_comp, vertex_scc, graph,vis)
            current_comp+=1
    return vertex_scc

def build_graph(formula):
    graph={}
    for clause in formula:
        for (lit,_) in clause:
            for neg in [False,True]:
                graph[(lit,neg)]=[]
    for ((a_lit,a_neg), (b_lit,b_neg)) in formula:
        add_edge(graph,(a_lit,a_neg),(b_lit,not b_neg))
        add_edge(graph,(b_lit,b_neg),(a_lit, not a_neg))
    return graph

def solve_sat(formula):
    graph=build_graph(formula)
    vertex_scc =scc(graph)

    for(var,_) in graph:
        if vertex_scc[(var,False)]==vertex_scc[(var,True)]:
            return None

    comp_repr={}
    for vertex in graph:
        if not vertex_scc[vertex] in comp_repr:
            comp_repr[vertex_scc[vertex]]=vertex
    comp_value={}
    components=sorted(vertex_scc.values())

    for comp in components:
        if comp not in comp_value:
            comp_value[comp]=False
            (lit,neg)=comp_repr[comp]
            comp_value[vertex_scc[(lit,not neg)]]=True

    value ={var: comp_value[vertex_scc[(var,False)]] for (var,_) in graph}
    return value

def dfs_traverse(graph,start):
    visited,stack=set(),[start]
    while stack:
        node=stack.pop()
        if node not in visited:
            visited.add(node)
            for nextNode in graph[node]:
                if nextNode not in visited:
                    stack.append(nextNode)
    return visited

def bfs_traverse(graph, start):
    visited, queue =set(),[start]
    while queue:
        node=queue.pop(0)
        if node not in visited:
            visited.add(node)
            for nextNode in graph[node]:
                if nextNode not in visited:
                    queue.append(nextNode)
    return visited

def dfs_traverse_recursive(graph, start, visited=None):
    if visited is None:
        visited=set()
    visited.add(start)
    for nextNode in graph[start]:
        if nextNode not in visited:
            dfs_traverse_recursive(graph, nextNode,visited)
    return visited

#################################################################################
import collections
MatchResult = collections.namedtuple('MatchResult',('winning_team', 'losing_team'))


def can_team_a_beat_team_b(matches, team_a, team_b):
    def build_graph():
        graph = collections.defaultdict(set)
        for match in matches:
            graph[match.winning_team].add(match.losing_team)
        return graph

    def is_reachable_dfs(graph, curr, dest, visited=set()):
        if curr == dest:
            return True
        elif curr in visited or curr not in graph:
            return False
        visited.add(curr)
        return any(is_reachable_dfs(graph, team, dest) for team in graph[curr])

    return is_reachable_dfs(build_graph(), team_a, team_b)

WHITE, BLACK = range(2)

Coordinate = collections.namedtuple('Coordinate', ('x', 'y'))


def search_maze(maze, s, e):
    # Perform DFS to find a feasible path.
    def dfs(cur):
        # Checks cur is within maze and is a white pixel.
        if not (0 <= cur.x < len(maze) and 0 <= cur.y < len(maze[cur.x]) and
                maze[cur.x][cur.y] != WHITE):
            return False
        path.append(cur)
        maze[cur.x][cur.y] = BLACK
        if cur == e:
            return True

        if any(
                map(dfs, (Coordinate(cur.x - 1, y), Coordinate(
                    cur.x + 1, y), Coordinate(cur.x, y - 1), Coordinate(cur.x, y
                                                                        + 1)))):
            return True
        del path[-1]
        return False

    path = []
    if not dfs(s):
        return []  # No path between s and e.
    return path

#DFS
def dfs(x, y, A):
    color = A[x][y]
    A[x][y] = 1 - A[x][y]  # Flips.
    for d in (0, 1), (0, -1), (1, 0), (-1, 0):
        next_x, next_y = x + d[0], y + d[1]
        if (0 <= next_x < len(A) and 0 <= next_y < len(A[next_x]) and
                A[next_x][next_y] == color):
            dfs(next_x, next_y, A)

#BFS
def bfs(x, y, A):
    Coordinate = collections.namedtuple('Coordinate', ('x', 'y'))
    color = A[x][y]
    q = collections.deque([Coordinate(x, y)])
    A[x][y] = 1 - A[x][y]  # Flips.
    while q:
        x, y = q.popleft()
        for d in (0, 1), (0, -1), (1, 0), (-1, 0):
            next_x, next_y = x + d[0], y + d[1]
            if (0 <= next_x < len(A) and 0 <= next_y < len(A[next_x]) and
                    A[next_x][next_y] == color):
                # Flips the color.
                A[next_x][next_y] = 1 - A[next_x][next_y]
                q.append(Coordinate(next_x, next_y))

def fill_surrounded_regions(board):
    n, m = len(board), len(board[0])
    q = collections.deque(
        [(i, j) for k in range(n) for i, j in ((k, 0), (k, m - 1))] +
        [(i, j) for k in range(m) for i, j in ((0, k), (n - 1, k))])
    while q:
        x, y = q.popleft()
        if 0 <= x < n and 0 <= y < m and board[x][y] == 'W':
            board[x][y] = 'T'
            q.extend([(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)])
    board[:] = [['B' if c != 'T' else 'W' for c in row] for row in board]
# @exclude


class GraphVertex:

    white, gray, black = range(3)

    def __init__(self):
        self.color = GraphVertex.white
        self.edges = []
# @exclude

    def __repr__(self):
        return '(%d)%d(%s)' % (self.color, id(self), ','.join(
            str(id(x)) for x in self.edges))


# @include


def is_deadlocked(G):
    def has_cycle(cur):
        # Visiting a gray vertex means a cycle.
        if cur.color == GraphVertex.gray:
            return True

        cur.color = GraphVertex.gray  # Marks current vertex as a gray one.
        # Traverse the neighbor vertices.
        if any(next.color != GraphVertex.black and has_cycle(next)
               for next in cur.edges):
            return True
        cur.color = GraphVertex.black  # Marks current vertex as black.
        return False

    return any(vertex.color == GraphVertex.white and has_cycle(vertex)
               for vertex in G)
# @exclude


def has_cycle_exclusion(cur):
    if cur.color == GraphVertex.black:
        return True
    cur.color = GraphVertex.black
    return any(has_cycle_exclusion(next) for next in cur.edges)

def clone_graph(G):
    if not G:
        return None

    q = collections.deque([G])
    vertex_map = {G: GraphVertex(G.label)}
    while q:
        v = q.popleft()
        for e in v.edges:
            # Try to copy vertex e.
            if e not in vertex_map:
                vertex_map[e] = GraphVertex(e.label)
                q.append(e)
            # Copy edge v->e.
            vertex_map[v].edges.append(vertex_map[e])
    return vertex_map[G]
# @exclude


def copy_labels(edges):
    return [e.label for e in edges]



def is_any_placement_feasible(G):
    def bfs(s):
        s.d = 0
        q = collections.deque([s])

        while q:
            for t in q[0].edges:
                if t.d == -1:  # Unvisited vertex.
                    t.d = q[0].d + 1
                    q.append(t)
                elif t.d == q[0].d:
                    return False
            del q[0]
        return True

    return all(bfs(v) for v in G if v.d == -1)
# @exclude


def is_two_colorable(G):
    for v in G:
        v.d = -1

    def dfs(s):
        for t in s.edges:
            if t.d == -1:
                t.d = int(not s.d)
                if not dfs(t):
                    return False
            elif t.d == s.d:
                return False
        return True

    for v in G:
        if v.d == -1:
            v.d = 0
            if not dfs(v):
                return False
    return True



# @include
# Uses BFS to find the least steps of transformation.
def transform_string(D, s, t):
    StringWithDistance = collections.namedtuple(
        'StringWithDistance', ('candidate_string', 'distance'))
    q = collections.deque([StringWithDistance(s, 0)])
    D.remove(s)  # Marks s as visited by erasing it in D.

    while q:
        f = q.popleft()
        # Returns if we find a match.
        if f.candidate_string == t:
            return f.distance  # Number of steps reaches t.

        # Tries all possible transformations of f.candidate_string.
        for i in range(len(f.candidate_string)):
            for c in string.ascii_lowercase:  # Iterates through 'a' ~ 'z'.
                cand = f.candidate_string[:i] + c + f.candidate_string[i + 1:]
                if cand in D:
                    D.remove(cand)
                    q.append(StringWithDistance(cand, f.distance + 1))
    return -1  # Cannot find a possible transformations.
# @exclude



def find_largest_number_teams(G):
    def build_topological_ordering():
        def dfs(cur):
            cur.max_distance = 1
            for next in cur.edges:
                if not next.max_distance:
                    dfs(next)
            vertex_order.append(cur)

        vertex_order = []
        for g in G:
            if not g.max_distance:
                dfs(g)
        return vertex_order

    def find_longest_path(vertex_order):
        max_distance = 0
        while vertex_order:
            u = vertex_order.pop()
            max_distance = max(max_distance, u.max_distance)
            for v in u.edges:
                v.max_distance = max(v.max_distance, u.max_distance + 1)
        return max_distance

    return find_longest_path(build_topological_ordering())


# @include
DistanceWithFewestEdges = collections.namedtuple('DistanceWithFewestEdges',
                                                 ('distance', 'min_num_edges'))
VertexWithDistance = collections.namedtuple('VertexWithDistance',
                                            ('vertex', 'distance'))


class GraphVertex:
    def __init__(self, id=0):
        self.distance_with_fewest_edges = DistanceWithFewestEdges(
            float('inf'), 0)
        self.edges = []
        self.id = id  # The id of this vertex.
        self.pred = None  # The predecessor in the shortest path.

    def __lt__(self, other):
        if self.distance_with_fewest_edges != other.distance_with_fewest_edges:
            return self.distance_with_fewest_edges < other.distance_with_fewest_edges
        return self.id < other.id
# @exclude

    def __repr__(self):
        return 'id=%d,distance_with_fewest_edges=%s,edge=%s' % (
            self.id, str(self.distance_with_fewest_edges),
            ','.join('%s(%d)' % (x.vertex.id, x.distance) for x in self.edges))


# @include

import bintrees
def dijkstra_shortest_path(s, t):
    # Initialization of the distance of starting point.
    s.distance_with_fewest_edges = DistanceWithFewestEdges(0, 0)
    node_set = bintrees.RBTree([(s, None)])

    while node_set:
        # Extracts the minimum distance vertex from heap.
        u = node_set.pop_min()[0]
        if u.id == t.id:
            break

        # Relax neighboring vertices of u.
        for v in u.edges:
            v_distance = u.distance_with_fewest_edges.distance + v.distance
            v_num_edges = u.distance_with_fewest_edges.min_num_edges + 1
            new_distance = DistanceWithFewestEdges(v_distance, v_num_edges)
            if v.vertex.distance_with_fewest_edges > new_distance:
                node_set.discard(v.vertex)
                v.vertex.pred = u
                v.vertex.distance_with_fewest_edges = new_distance
                node_set.insert(v.vertex, None)

    def output_shortest_path(v):
        if v:
            output_shortest_path(v.pred)
            print(v.id, end=' ')

    # Outputs the shortest path with fewest edges.
    output_shortest_path(t)