def get_factors(n):
    def factor(n,i,combi,res):
        while i*i<=n:
            if n%i==0:
                res+=combi+[i,int(n/i)]
                factor(n/i,i,combi+[i],res)
            i+=1
        return res
    return factor(n,2,[],[])

def get_factors_iterative1(n):
    todo,res=[(n,2,[])],[]
    while todo:
        n,i,combi=todo.pop()
        while i*i<=n:
            if n%i==0:
                res+=combi+[i,n/i]
                todo+=(n/i,i,combi+[i])
            i+=1
    return res

def get_factors_interatives2(n):
    ans,stack,x=[],[],2
    while True:
        if x>n/x:
            if not stack:
                return ans
            ans.append(stack+[n])
            x=stack.pop()
            n*=x
            x+=1
        elif n%x==0:
            stack.append(x)
            n/=x
        else:
            x+=1

print(get_factors(32))

def num_islands(grid):
    count =0
    for i,row in enumerate(grid):
        for j, col in enumerate(grid[i]):
            if col=='1':
                DFS(grid,i,j)
                count+=1
    return count

def DFS(grid,i,j):
    if (i<0 or i>=len(grid)) or (j<0 or len(grid[0])):
        return
    if grid[i][j]!='1':
        return
    grid[i][j]='0'
    DFS(grid,i+1,j)
    DFS(grid,i-1,j)
    DFS(grid,i,j+1)
    DFS(grid,i,j-1)

def pacific_atlantic(matrix):
    n=len(matrix)
    if not n: return []
    m=len(matrix[0])
    if not m: return []
    res=[]
    atlantic=[[False for _ in range(n)] for _ in range(m)]
    pacific = [[False for _ in range(n)] for _ in range(m)]
    for i in range(n):
        DFS2(pacific, matrix, float("-inf"),i,0)
        DFS2(atlantic, matrix, float("-inf"),i,m-1)
    for i in range(m):
        DFS2(pacific, matrix, float("-inf"), 0, i)
        DFS2(atlantic, matrix, float("-inf"), n-1, i)
    for i in range(n):
        for j in range(m):
            if pacific[i][j] and atlantic[i][j]:
                res.append([i,j])
    return res

def DFS2(grid,matrix,height,i,j):
    if i<0 or i>=len(matrix) or j<0 or j>=len(matrix[0]):
        return
    if grid[i][j] or matrix[i][j] < height:
        return
    grid[i][j]= True

    DFS2(grid,matrix,matrix[i][j],i-1,j)
    DFS2(grid,matrix,matrix[i][j],i+1,j)
    DFS2(grid,matrix,matrix[i][j],i,j-1)
    DFS2(grid,matrix,matrix[i][j],i,j+1)


def solveSudoku(self,board):
    self.board=board
    self.val=self.PossibleVals()
    self.Solver()

def PossibleVals(self):
    a="123456789"
    d,val={},{}
    for i in range(9):
        for j in range(9):
            ele=self.board[i][j]
            if ele!=".":
                d[("r",i)]=d.get(("r",i),[])+[ele]
                d[("c", i)] = d.get(("c", i), []) + [ele]
                d[(i//3,j//3)]=d.get((i//3,j//3),[]) + [ele]
            else:
                val[(i,j)]=[]
    for (i,j) in val.keys():
        inval=d.get(("r",i),[]) + d.get(("c",j),[])+d.get((i/3,j/3),[])
        val[(i,j)]=[n for n in a if n not in inval]
    return val
def Solver(self):
    if len(self.val)==0:
        return True
    kee=min(self.val.keys(),key=lambda  x:len(self.val[x]))
    nums=self.val[kee]
    for n in nums:
        update ={kee:self.val[kee]}
        if self.ValidOne(n,kee,update):
            if self.Solver():
                return True
            self.undo(kee,update)
    return False

def ValidOne(self,n,kee,update):
    self.board[kee[0]][kee[1]]=n
    del self.val[kee]
    i,j=kee
    for ind in self.val.keys():
        if n in self.val[ind]:
            if ind[0]==i or ind[1]==j or (ind[0]/3,ind[1]/3)==(i/3,j/3):
                update[ind]=n
                self.val[ind].remove(n)
                if len(self.val[ind])==0:
                    return False
    return True


def undo(self,kee,update):
    self.board[kee[0]][kee[1]]="."
    for k in update:
        if k not in self.val:
            self.val[k]=update[k]
        else:
            self.val[k].appenjd(update[k])
    return None

def walls_and_gates(rooms):
    for i in range(len(rooms)):
        for j in range(len(rooms[0])):
            if rooms[i][j]==0:
                DFS3(rooms,i,j,0)
def DFS3(rooms,i,j,depth):
    if (i<0 or i>=len(rooms)) or (j<0 or j>=len(rooms[0])):
        return
    if rooms[i][j]<depth:
        return
    rooms[i][j]=depth
    DFS3(rooms,i+1,j,depth+1)
    DFS3(rooms, i - 1, j, depth + 1)
    DFS3(rooms, i, j+1, depth + 1)
    DFS3(rooms, i, j-1, depth + 1)








