from collections import deque
import re
numeric_value = re.compile('\d+(\.\d+)?')
__operators__="+-/*"
__parenthesis__="()"
__priority__={
    '+':0,
    '-':0,
    '*':1,
    '/':1
}

def is_operator(token):
    return token in __operators__

def higher_priority(op1,op2):
    return __priority__[op1]>=__priority__[op2]

def calc(n2,n1,operator):
    if operator=='-': return n1-n2
    elif operator=='+': return n1+n2
    elif operator=='*': return n1*n2
    elif operator=='/': return n1/n2
    return 0

def apply_operation(op_stack, out_stack):
    out_stack.append(calc(out_stack.pop(), out_stack.pop(), op_stack.pop()))

def parse(expression):
    result =[]
    current =""
    for i in expression:
        if i.isdigit() or i=='.':
            current+=i
        else:
            if len(current)>0:
                result.append(current)
                current=""
            if i!=' ':
                result.append(i)

    if len(current)>0:
        result.append(current)
    return result

def evaluate(expression):

    op_stack= deque()
    out_stack=deque()
    for token in parse(expression):
        if numeric_value.match(token):
            out_stack.append(float(token))
        elif token=="(":
            op_stack.append(token)
        elif token==')':
            while len(op_stack)>0 and op_stack[-1]!='(':
                apply_operation(op_stack,out_stack)
            op_stack.pop()
        else:
            while len(op_stack)>0 and is_operator(op_stack[-1]) and higher_priority(op_stack[-1],token):
                apply_operation(op_stack,out_stack)
            op_stack.append(token)
    while len(op_stack)>0:
        apply_operation(op_stack,out_stack)
    return out_stack[-1]


