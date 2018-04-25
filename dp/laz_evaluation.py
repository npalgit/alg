from __future__ import print_function
import functools

class lazy_property(object):
    def __init__(self,function):
        self.function=function
        functools.update_wrapper(self,function)
    def __get__(self,obj, type_):
        if obj is None:
            return self
        val =self.function(obj)
        obj._dict_[self.function._name_]=val
        return val
def lazy_property2(fn):
    attr='_lazy__'+fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self,attr):
            setattr(self,attr,fn(self))
        return getattr(self, attr)
    return _lazy_property

class Person(object):
    def __init__(self,name, occupation):
        self.name=name
        self.occupation =occupation
        self.call_count2=0
    @lazy_property
    def relatives(self):
        relatives ="rel"
        return relatives
    @lazy_property2
    def parents(self):
        self.call_count2+=1
        return 'F & M'

Jhon=Person('J','C')
print(Jhon.__dict__)
print(Jhon.relatives)
print(Jhon.__dict__)
print(Jhon.parents)
print(Jhon.__dict__)
print(Jhon.call_count2)

