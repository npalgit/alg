def extended_gcd(a,b):
    old_s,s=1,0
    old_t,t=0,1
    old_r,r=a,b
    while r!=0:
        quetient = old_r/r
        old_r, r = r, old_r - quetient*r
        old_s, s = s, old_s - quetient*s
        old_t, t = t, old_t - quetient*t
    return old_s, old_t, old_r

def gcd(a,b):
    while True:
        if b==0:
            return a
        a,b=b,a%b

def lcm(a,b):
    return a*b/gcd(a,b)


def gen_strobogrammatic(n):
    result = helper(n,n)
    return  result

def helper(n,length):
    if n==0:
        return [""]
    if n==1:
        return ["1","0","8"]

    middles = helper(n-2, length)
    result=[]
    for middle in middles:
        if n!= length:
            result.append("0"+middle+"0")
        result.append("8" + middle + "8")
        result.append("1" + middle + "1")
        result.append("9" + middle + "6")
        result.append("6" + middle + "9")
    return result


def strobogrammaticInRange(low, high):
    res=[]
    count =0
    for i in range(len(low), len(high)+1):
        res.extend(helper2(i,i))
    for perm in res:
        if len(perm)==len(low) and int(perm) <int(low):
            continue
        elif len(perm)==len(high) and int(perm)>int(high):
            continue
        else:
            count+=1
    return count

def helper2(n,length):
    if n==0:
        return [""]
    if n==1:
        return ["0","8","1"]

    middles = helper2(n-2, length)
    result=[]
    for middle in middles:
        if n!= length:
            result.append("0"+middle+"0")
        result.append("1" + middle + "1")
        result.append("6" + middle + "9")
        result.append("9" + middle + "6")
        result.append("8" + middle + "8")
    return result


def is_strobogrammatic(num):
    comb ="00 11 88 69 96"
    i=0
    j=len(num)-1
    while i<=j:
        x=comb.find(num[i]+num[j])
        if x==-1:
            return False
        i+=1
        j-=1
    return True

def find_nth_digit(n):
    len =1
    count =9
    start =1
    while n> len * count:
        n-=len*count
        len+=1
        count*=10
        start*=10
    start +=(n-1)/len
    s=str(start)
    return int(s[(n-1)%len])

def prime_test(n):
    if n<=1:
        return False
    if n==2 or n==3:
        return True
    if n%2==0 or n%3==0:
        return False
    j=5
    while (j*j<=n):
        if n%(j)==0 or n%(j+2)==0:
            return False
        j+=6
    return True

def primes(x):
    assert(x>=0)
    sieve_size=(x//2-1) if x%2 ==0 else (x//2)
    sieve =[1 for v in range(sieve_size)]
    primes=[]
    if x>=2:
        primes.append(2)
    for i in range(0,sieve_size):
        if sieve[i]==1:
            value_at_i=i*2+3
            primes.append(value_at_i)
            for j in range(i,sieve_size,value_at_i):
                sieve[j]=0
    return primes

def pythagoras(opposite, adjacent, hypotenuse):
    try:
        if opposite==str("?"):
            return ("Opposite = " + str(((hypotenuse**2-adjacent**2))**0.5))
        elif adjacent==str("?"):
            return ("Adjacent = " + str(((hypotenuse**2-opposite**2))**0.5))
        elif hypotenuse == str("?"):
            return ("Hypotenuse = " + str(((opposite ** 2 - adjacent ** 2)) ** 0.5))
        else:
            return "You already know the answer!"
    except:
        print("Error")

import random, sys
def pow2_factor(n):
    power =0
    while n%2 ==0:
        n/=2
        power+=1
    return power,n

def is_prime(n,k):
    r,d=pow2_factor(n-1)

    def valid_witness(a):
        x = pow(a,d,n)
        if x==1 or x==n-1:
            return False

        for _ in range(r-1):
            x = pow(x,2,n)

            if x==1:
                return True
            if x== n-1:
                return False
        return True

    for _ in range(k):
        if valid_witness(random.randrange(2,n-2)):
            return False
    return True


import random

def genprime(k):
    while True:
        n = random.randrange(2**(k-1), 2**k)
        if is_prime(n,128):
            return n
def modinv(a,m):
    x,y,g=extended_gcd(a,m)
    return x %m

def generate_key(k):
    p_size = k/2
    q_size =k-p_size
    e=genprime(k)

    while True:
        p=genprime(k/2)
        if p%e!=1:
            break
    while True:
        q=genprime(k-k/2)
        if q%e !=1:
            break

    n =p*q
    l = (p-1)*(q-1)
    d=modinv(e,l)
    return n,e,d

def square_root(n,p):
    guess = float(n)/2
    while abs(guess*guess-n)>p:
        guess = (guess+(n/guess))/2
    return guess
