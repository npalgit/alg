import threading
class Semaphore():
    def __init__(self, capacity):
        self.cv=threading.Condition()
        self.capacity=capacity
        self.taken=0

    def p(self):
        self.cv.acquire()
        while(self.taken==self.capacity):
            self.cv.wait() #wait
        self.taken+=1
        self.cv.release()
    def v(self):
        self.cv.acquire()
        self.taken-=1
        self.cv.notify() #ready
        self.cv.release()
    def num_take(self):
        return self.taken

class Worker(threading.Thread): #inherit
    def __init__(self,name, semaphore):
        threading.Thread.__init__(self) #init
        self.semaphore =semaphore
        self.name=name

    def run(self):
        while True:
            self.semaphore.p()
            self.semaphore.v()

Worker("n").start()

class ServiceThread(threading.Thread):
    lock=threading.Lock()
    def __init__(self,data):
        super(ServiceThread, self).__init__(())
        self.data=data

    def run(self):
        with ServiceThread.lock:
            print("do sth")

ServiceThread(123).start()

ServiceThread(456).join()

import concurrent.futures
def func(arg):
    print("123")
data=234
executor=concurrent.futures.ThreadPoolExecutor(max_workers=4)
executor.submit(func, data)

#difference
semaphore=threading.Condition
mutex=threading.Lock

