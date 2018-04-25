# @include
import threading


class Semaphore():

    def __init__(self, max_available):
        self.cv = threading.Condition() #mutex
        self.MAX_AVAILABLE = max_available
        self.taken = 0

    def acquire(self):
        self.cv.acquire()
        while (self.taken == self.MAX_AVAILABLE):
            self.cv.wait()
        self.taken += 1
        self.cv.release()

    def release(self):
        self.cv.acquire()
        self.taken -= 1
        self.cv.notify()
        self.cv.release()

    # @exclude
    def num_taken(self):
        return self.taken

import time
import random


class Worker(threading.Thread):

    def __init__(self, name, semaphore):
        threading.Thread.__init__(self)
        self.semaphore = semaphore
        self.name = name

    def run(self):
        while True:
            self.semaphore.acquire()
            print(self.name +
                  " has acquired semaphore, the number taken is  " +
                  str(self.semaphore.taken) +
                  "\n")
            rnd_sleep = random.randint(200, 600) / 1000.0
            time.sleep(rnd_sleep)
            print(self.name + " is about to release acquired semaphore\n")
            self.semaphore.release()
            rnd_sleep = random.randint(2000, 6000) / 1000.0
            time.sleep(rnd_sleep)

if __name__ == '__main__':
    S = Semaphore(9)
    S.acquire()
    S.acquire()
    S.release()
    S.release()

    for i in range(10):
        w = Worker("worker-" + str(i), S)
        w.start()
import time
import threading


# @include
class SpellCheckService:
    w_last = closest_to_last_word = None

    @staticmethod
    def service(req, resp):
        w = req.extract_word_to_check_from_request()
        if w != SpellCheckService.w_last:
            SpellCheckService.w_last = w
            SpellCheckService.closest_to_last_word = closest_in_dictionary(w)
        resp.encode_into_response(SpellCheckService.closest_to_last_word)
# @exclude


class ServiceRequest:

    def __init__(self, s):
        self.request = s

    def extract_word_to_check_from_request(self):
        return self.request


class ServiceResponse:
    response = None

    def encode_into_response(self, s):
        self.response = s


def closest_in_dictionary(w):
    time.sleep(0.2)
    return [w + '_result']


class ServiceThread(threading.Thread):

    lock = threading.Lock()

    def __init__(self, data):
        super().__init__()
        self.data = data

    def run(self):
        start_time = time.time()
        req = ServiceRequest(self.data)
        resp = ServiceResponse()
        with ServiceThread.lock:
            SpellCheckService.service(req, resp)
            print(self.data, '->', resp.response, '(%.3f sec)' %
                  (time.time() - start_time))


def main():
    i = 0
    while True:
        ServiceThread('req:%d' % (i + 1)).start()
        if i > 0:
            # while req:i+1 is computed we could return req:i from the cache
            ServiceThread('req:%d' % i).start()
        time.sleep(0.5)
        i += 1


if __name__ == '__main__':
    main()
import time
import threading


# @include
class SpellCheckService:
    w_last = closest_to_last_word = None
    lock = threading.Lock()

    @staticmethod
    def service(req, resp):
        w = req.extract_word_to_check_from_request()
        result = None
        with SpellCheckService.lock:
            if w == SpellCheckService.w_last:
                result = SpellCheckService.closest_to_last_word.copy()
        if result is None:
            result = closest_in_dictionary(w)
            with SpellCheckService.lock:
                SpellCheckService.w_last = w
                SpellCheckService.closest_to_last_word = result
        resp.encode_into_response(result)
# @exclude


class ServiceRequest:

    def __init__(self, s):
        self.request = s

    def extract_word_to_check_from_request(self):
        return self.request


class ServiceResponse:
    response = None

    def encode_into_response(self, s):
        self.response = s


def closest_in_dictionary(w):
    time.sleep(0.2)
    return [w + '_result']


class ServiceThread(threading.Thread):

    def __init__(self, data):
        super().__init__()
        self.data = data

    def run(self):
        start_time = time.time()
        req = ServiceRequest(self.data)
        resp = ServiceResponse()
        SpellCheckService.service(req, resp)
        print(self.data, '->', resp.response, '(%.3f sec)' %
              (time.time() - start_time))


def main():
    i = 0
    while True:
        ServiceThread('req:%d' % (i + 1)).start()
        if i > 0:
            # while req:i+1 is computed we could return req:i from the cache
            ServiceThread('req:%d' % i).start()
        time.sleep(0.5)
        i += 1


if __name__ == '__main__':
    main()

import sys
import threading

# @include
N = 1000000
counter = 0


def increment_thread():
    global counter
    for _ in range(N):
        counter = counter + 1

t1 = threading.Thread(target=increment_thread)
t2 = threading.Thread(target=increment_thread)

t1.start()
t2.start()
t1.join()
t2.join()

print(counter)
# @exclude

import threading


# @include
class OddEvenMonitor(threading.Condition):

    ODD_TURN = True
    EVEN_TURN = False

    def __init__(self):
        super().__init__()
        self.turn = self.ODD_TURN

    def wait_turn(self, old_turn):
        with self:
            while self.turn != old_turn:
                self.wait()

    def toggle_turn(self):
        with self:
            self.turn ^= True
            self.notify()


class OddThread(threading.Thread):

    def __init__(self, monitor):
        super().__init__()
        self.monitor = monitor

    def run(self):
        for i in range(1, 101, 2):
            self.monitor.wait_turn(OddEvenMonitor.ODD_TURN)
            print(i)
            self.monitor.toggle_turn()


class EvenThread(threading.Thread):

    def __init__(self, monitor):
        super().__init__()
        self.monitor = monitor

    def run(self):
        for i in range(2, 101, 2):
            self.monitor.wait_turn(OddEvenMonitor.EVEN_TURN)
            print(i)
            self.monitor.toggle_turn()
# @exclude


def main():
    monitor = OddEvenMonitor()
    t1 = OddThread(monitor)
    t2 = EvenThread(monitor)
    t1.start()
    t2.start()
    t1.join()
    t2.join()


if __name__ == '__main__':
    main()

import socket


def process_req(sock):
    while True:
        data = sock.recv(1024)
        if not data:
            break
        print(data)
        sock.sendall(data)


# @include
SERVERPORT = 8080


def main():
    serversock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversock.bind(('', SERVERPORT))
    serversock.listen(5)
    while True:
        sock, addr = serversock.accept()
        process_req(sock)
# @exclude

if __name__ == '__main__':
    main()

import socket
import threading


def process_req(sock):
    while True:
        data = sock.recv(1024)
        if not data:
            break
        print(data)
        sock.sendall(data)


# @include
SERVERPORT = 8080


serversock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serversock.bind(('', SERVERPORT))
serversock.listen(5)
while True:
    sock, addr = serversock.accept()
    threading.Thread(target=process_req, args=(sock, )).start()
# @exclude

import socket
import concurrent.futures


def process_req(sock):
    while True:
        data = sock.recv(1024)
        if not data:
            break
        print(data)
        sock.sendall(data)


# @include
SERVERPORT = 8080
NTHREADS = 2


executor = concurrent.futures.ThreadPoolExecutor(max_workers=NTHREADS)
serversock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serversock.bind(('', SERVERPORT))
serversock.listen(5)
while True:
    sock, addr = serversock.accept()
    executor.submit(process_req, sock)
# @exclude

import time
import threading


# @include
class Account:

    _global_id = 0

    def __init__(self, balance):
        self._balance = balance
        self._id = Account._global_id
        Account._global_id += 1
        self._lock = threading.RLock() #rlock

    def get_balance(self):
        return self._balance

    @staticmethod
    def transfer(acc_from, acc_to, amount):
        th = threading.Thread(target=acc_from._move, args=(acc_to, amount))
        th.start()

    def _move(self, acc_to, amount):
        with self._lock:
            if amount > self._balance:
                return False
            acc_to._balance += amount
            self._balance -= amount
            print('returning True')
            return True
# @exclude


def main():
    acc_from = Account(200)
    acc_to = Account(100)
    print('initial balances =', acc_from.get_balance(), acc_to.get_balance())
    assert (acc_from.get_balance(), acc_to.get_balance()) == (200, 100)
    Account.transfer(acc_from, acc_to, 50)
    time.sleep(0.1)
    print('new balances =', acc_from.get_balance(), acc_to.get_balance())
    assert (acc_from.get_balance(), acc_to.get_balance()) == (150, 150)


if __name__ == '__main__':
    main()

import time
import threading


class Account:

    _global_id = 0

    def __init__(self, balance):
        self._balance = balance
        self._id = Account._global_id
        Account._global_id += 1
        self._lock = threading.RLock()

    def get_balance(self):
        return self._balance

    @staticmethod
    def transfer(acc_from, acc_to, amount):
        th = threading.Thread(target=acc_from._move, args=(acc_to, amount))
        th.start()

    def _move(self, acc_to, amount):
        # @include
        lock1 = self._lock if self._id < acc_to._id else acc_to._lock
        lock2 = acc_to._lock if self._id < acc_to._id else self._lock
        # Does not matter if lock1 equals lock2: since recursive_mutex locks
        # are reentrant, we will re-acquire lock2.
        with lock1, lock2:
            # @exclude
            if amount > self._balance:
                return False
            acc_to._balance += amount
            self._balance -= amount
            print('returning True')
            return True


def main():
    acc_from = Account(200)
    acc_to = Account(100)
    print('initial balances =', acc_from.get_balance(), acc_to.get_balance())
    Account.transfer(acc_from, acc_to, 50)
    assert (acc_from.get_balance(), acc_to.get_balance()) == (150, 150)
    time.sleep(0.1)
    print('new balances =', acc_from.get_balance(), acc_to.get_balance())
    Account.transfer(acc_from, acc_from, 50)
    assert (acc_from.get_balance(), acc_to.get_balance()) == (150, 150)
    time.sleep(0.1)
    print('new balances =', acc_from.get_balance(), acc_to.get_balance())
    assert (acc_from.get_balance(), acc_to.get_balance()) == (150, 150)


if __name__ == '__main__':
    main()
import time
import random
import threading


def do_something_else():
    time.sleep(random.random())


# @include
# LR and LW are class attributes in the RW class.
# They serve as read and write locks. The integer
# variable read_count in RW tracks the number of readers.
class Reader(threading.Thread):
    # @exclude

    def __init__(self, name):
        super().__init__(name=name, daemon=True)
# @include

    def run(self):
        while True:
            with RW.LR:
                RW.read_count += 1

# @exclude
            print('Reader', self.name, 'is about to read')
            # @include
            print(RW.data)
            with RW.LR:
                RW.read_count -= 1
                RW.LR.notify()
            do_something_else()


class Writer(threading.Thread):
    # @exclude

    def __init__(self, name):
        super().__init__(name=name, daemon=True)
# @include

    def run(self):
        while True:
            with RW.LW:
                done = False
                while not done:
                    with RW.LR:
                        if RW.read_count == 0:
                            # @exclude
                            print('Writer', self.name, 'is about to write')
                            # @include
                            RW.data += 1
                            done = True
                        else:
                            # use wait/notify to avoid busy waiting
                            while RW.read_count != 0:
                                RW.LR.wait()
            do_something_else()
# @exclude


class RW:
    data = 0
    LR = threading.Condition()
    read_count = 0
    LW = threading.Lock()


def main():
    r0 = Reader('r0')
    r1 = Reader('r1')
    w0 = Writer('w0')
    w1 = Writer('w1')
    r0.start()
    r1.start()
    w0.start()
    w1.start()
    time.sleep(10)


if __name__ == '__main__':
    main()

import sys
import time
import concurrent.futures


# @include
# Performs basic unit of work
def worker(lower, upper):
    for i in range(lower, upper + 1):
        assert collatz_check(i, set())
    print('(%d,%d)' % (lower, upper))


# @exclude


# @include
# Checks an individual number
def collatz_check(x, visited):
    if x == 1:
        return True
    elif x in visited:
        return False
    visited.add(x)
    if x & 1:  # odd number
        return collatz_check(3 * x + 1, visited)
    else:  # even number
        return collatz_check(x >> 1, visited)  # divide by 2
# @exclude


def main():
    N = 10000000
    RANGESIZE = 1000000
    NTHREADS = 4
    if len(sys.argv) > 1:
        N = int(sys.argv[1])
    if len(sys.argv) > 2:
        RANGESIZE = int(sys.argv[2])
    if len(sys.argv) > 3:
        NTHREADS = int(sys.argv[3])

    assert collatz_check(1, set())
    assert collatz_check(3, set())
    assert collatz_check(8, set())
    start_time = time.time()

    # @include
    # Uses the library thread pool for task assignment and load balancing
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=NTHREADS)
    with executor:
        for i in range(N // RANGESIZE):
            executor.submit(worker, i * RANGESIZE + 1, (i + 1) * RANGESIZE)
# @exclude
    print('Finished all threads')
    running_time = (time.time() - start_time) * 1000
    print('time in milliseconds for checking to %d is %d (%d per ms)' %
          (N, running_time, N / running_time))

if __name__ == '__main__':
    main()
