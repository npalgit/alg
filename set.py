class RandomizedSet():
    def __int__(self):
        pass
    def insert(self,val):
        pass
    def remove(self,val):
        pass
    def get_random(self):
        pass

from itertools import chain, combinations
def powerset(iterable):
    s=list(iterable)
    return chain.from_iterable(combinations(s,r) for r in range(len(s)+1))

def optimal_set_cover(universe, subsets, costs):
    pset =powerset(subsets.keys())
    best_set=None
    best_cost = float("inf")
    for subset in pset:
        covered = set()
        cost =0
        for s in subsets:
            covered.update(subsets[s])
            cost+=costs[s]
        if len(covered)==len(universe) and cost<best_cost:
            best_set=subset
            best_cost=cost
    return best_set

def greedy_set_cover(universe,subsets, costs):
    elements =set(e for s in subsets.keys() for e in subsets[s])
    if elements!=universe:
        return None
    covered =set()
    cover_sets=[]
    while covered != universe:
        min_cost_elem_ratio = float("inf")
        min_set =None

        for s, elements in subsets.items():
            new_elements =len(elements-covered)
            if new_elements!=0:
                cost_elem_ratio = costs[s]/new_elements
                if cost_elem_ratio < min_cost_elem_ratio:
                    min_cost_elem_ratio=cost_elem_ratio
                    min_set =s
        cover_sets.append(min_set)
        covered|=subsets[min_set]
    return cover_sets

