def numIslands2(m,n,positions):
    res=[]
    islands=Union()
    for p in map(tuple,positions):
        islands.add(p)
        for m in [(0,1),(0,-1),(1,0),(-1,0)]:
            q=p[0]+m[0], p[1]+m[1]
            if q in islands.id:
                islands.unite(p,q)
        res+=[islands.count]
    return res

class Union:
    def __init__(self):
        self.id={}
        self.sz={}
        self.count=0

    def add(self,p):
        self.id[p]=p
        self.sz[p]=1
        self.count+=1

    def root(self,i):
        while i!=self.id[i]:
            self.id[i]=self.id[self.id[i]]
            i=self.id[i]
        return i

    def unite(self,p,q):
        i,j=self.root(p),self.root(q)
        if i==j:
            return
        if self.sz[i]>self.sz[j]:
            i,j=j,i
        self.id[i]=j
        self.sz[j]+=self.sz[i]
        self.count-=1

positions = [[0,0], [0,1], [1,2], [2,1]]
print(numIslands2(3,3,positions))
