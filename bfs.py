def shortest_distance(grid):
    if not grid or not grid[0]:
        return -1
    matrix = [[[0,0] for i in range(len(grid[0]))] for j in range(len(grid))] #3d
    count=0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j]==1:
                bfs(grid,matrix,i,j,count)
                count+=1
    res=float('inf')
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j][1]==count:
                res=min(res,matrix[i][j][0])
    return res if res!=float('inf') else -1

def bfs(grid, matrix, i,j, count):
    q=[(i,j,0)]
    while q:
        i,j,step=q.pop(0) #or popleft
        for k,l in [(i-1,j), (i+1,j), (i,j-1),(i,j+1)]:
            if 0<=k < len(grid) and 0<=l<len(grid[0]) and matrix[k][l][1]==count and grid[k][l]==0:
                matrix[k][l][0]+=step+1
                matrix[k][l][1]=count+1
                q.append((k,l,step+1))


def ladderLength(beginWord, endWord, wordList):
    beginSet=set()
    endSet=set()
    beginSet.add(beginWord)
    endSet.add(endWord)
    result=2
    while len(beginSet)!=0 and len(endSet)!=0:
        if len(beginSet)>len(endSet):
            beginSet,endSet=endSet,beginSet
        nextBeginSet=set()
        for word in beginSet:
            for ladderWord in wordRange(word):
                if ladderWord in endSet:
                    return result
                if ladderWord in wordList:
                    nextBeginSet.add(ladderWord)
                    wordList.remove(ladderWord)
        beginSet=nextBeginSet
        result+=1
        print(beginSet)
        print(result)
    return 0
def wordRange(word):
    for ind in range(len(word)):
        tempC = word[ind]
        for c in [chr(x) for x in range(ord('a'),ord('z')+1)]:
            if c!= tempC:
                yield word[:ind]+c+word[ind+1:]
