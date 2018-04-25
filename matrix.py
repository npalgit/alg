def max_killed_enemies(grid):
    if not grid: return 0
    m,n=len(grid),len(grid[0])
    max_killed =0
    row_e, col_e=0,[0]*n
    for i in range(m):
        for j in range(n):
            if j==0 or grid[i][j-1]=='W':
                row_e=row_kills(grid,i,j)

            if i==0 or grid[i-1][j]=='W':
                col_e[j]=col_kills(grid,i,j)
            if grid[i][j]=='0':
                max_killed=max(max_killed, row_e+col_e[j])
    return max_killed

def row_kills(grid,i,j):
    num =0
    while j<len(grid[0]) and grid[i][j]!='W':
        if grid[i][j]=='E':
            num+=1
        j+=1
    return num

def col_kills(grid, i,j):
    num =0
    while i < len(grid) and grid[i][j]!='W':
        if grid[i][j]=='E':
            num+=1
        i+=1
    return num

def rotate_clockwise(matrix):
    new =[]
    for row in reversed(matrix):
        for i, elem in enumerate(row):
            try:
                new[i].append(elem)
            except IndexError:
                new.insert(i,[])
                new[i].append(elem)
    return new

def rotate_counterclockwise(matrix):
    new=[]
    for row in matrix:
        for i,elem in enumerate(reversed(row)):
            try:
                new[i].append(elem)
            except IndexError:
                new.insert(i,[])
                new[i].append(elem)
    return new

def top_left_invert(matrix):
    new =[]
    for row in matrix:
        for i,elem in enumerate(row):
            try:
                new[i].append(elem)
            except IndexError:
                new.insert(i,[])
                new[i].append(elem)
    return new

def bottom_left_invert(matrix):
    new =[]
    for row in reversed(matrix):
        for i, elem in enumerate(reversed(row)):
            try:
                new[i].append(elem)
            except IndexError:
                new.insert(i,[])
                new[i].append(elem)
    return new


def print_matrix(matrix, name):
    print('{}:\n['.format(name))
    for row in matrix:
        print('  {}'.format(row))
    print(']\n')

def count_paths(m,n):
    if m<1 or n<1:
        return -1
    count = [[None for j in range(n)] for i in range(m)]
    for i in range(n):
        count[0][i]=1
    for j in range(m):
        count[j][0]=1

    for i in range(1,m):
        for j in range(1,n):
            count[i][j]=count[i-1][j]+ count[i][j-1]
    print(count[m-1][n-1])


def rotate(mat):
    if not mat:
        return mat
    mat.reverse()
    for i in range(len(mat)):
        for j in range(i):
            mat[i][j], mat[j][i]=mat[j][i],mat[i][j]


def search_in_a_sorted_matrix(mat,m,n,key):
    i,j=m-1,0
    while i<=0 and j<n:
        if key ==mat[i][j]:
            print("found ")
            return

        if key < mat[i][j]:
            i-=1
        else:
            j+=1
    print("not found")

def multiply(A,B):
    if A is None or B is None: return None
    m,n,l=len(A),len(A[0]),len(B[0])
    if len(B)!=n:
        raise  Exception("A!=B")
    C =[[0 for _ in range(l)] for _ in range(m)]
    for i, row in enumerate(A):
        for k,eleA in enumerate(row):
            if eleA:
                for j,eleB in enumerate(B[k]):
                    if eleB: C[i][j] += eleA* eleB
    return C

def multiply2(A,B):
    if A is None or B is None: return None
    m,n,l=len(A),len(A[0]),len(B[0])
    if len(B)!=n:
        raise  Exception("A!=B")
    C =[[0 for _ in range(l)] for _ in range(m)]
    tableB={}
    for k, row in enumerate(B):
        tableB[k]={}
        for j,eleB in enumerate(row):
            if eleB: tableB[k][j]=eleB
    for i, row in enumerate(A):
        for k,eleA in enumerate(row):
            if eleA:
                for j,eleB in tableB[k].iteritems():
                    C[i][j]+=eleA*eleB
    return C

def multiply3(A,B):
    if A is None or B is None: return None
    m,n=len(A),len(A[0])
    if len(B)!=n:
        raise Exception("A!=B")
    l=len(B[0])
    table_A, table_B={},{}
    for i,row in enumerate(A):
        for j,ele in enumerate(row):
            if ele:
                if i not in table_A: table_A[i]={}
                table_A[i][j]=ele
    for i,row in enumerate(B):
        for j,ele in enumerate(row):
            if ele:
                if i not in table_B: table_B[i]={}
    C=[[0 for j in range(l)] for i in range(m)]
    for i in table_A:
        for k in table_A[i]:
            if k not in table_B: continue
            for j in table_B[k]:
                C[i][j]+=table_A[i][k]*table_B[k][j]
    return C

def spiral_traversal(matrix):
    res=[]
    if len(matrix)==0:
        return res

    row_begin=0
    row_end=len(matrix)-1
    col_begin=0
    col_end=len(matrix[0])-1

    while row_begin <=row_end and col_begin<=col_end:
        for i in range(col_begin,col_end+1):
            res.append(matrix[row_begin][i])
        row_begin+=1

        for i in range(row_begin,row_end+1):
            res.append(matrix[i][col_end])
        col_end-=1

        if row_begin <=row_end:
            for i in range(col_end, col_begin-1,-1):
                res.append(matrix[row_end][i])
        row_end-=1

        if col_begin<=col_end:
            for i in range(row_end,row_begin-1,-1):
                res.append(matrix[i][col_begin])
        col_begin+=1

    return res
