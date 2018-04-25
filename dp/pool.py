class ObjectPool(object):
    def __init__(self,queue,auto_get=False):
        self._queue=queue
        self.item=self._queue.get() if auto_get else None

    def __enter__(self):
        if self.item is None:
            self.item = self._queue.get()
        return self.item
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.item is not None:
            self._queue.put(self.item)
            self.item=None

    def __del__(self):
        if self.item is not None:
            self._queue.put(self.item)
            self.item=None

try:
    import queue
except ImportError:
    import Queue as queue

def test_object(queue):
    pool=ObjectPool(queue, True)
    print(pool.item) #get from pool


sample_queue=queue.Queue()
sample_queue.put('yaml')
with ObjectPool(sample_queue) as obj:
    print(obj)

print(sample_queue.get())

sample_queue.put('sam')
test_object(sample_queue)
print(sample_queue.get())
if not sample_queue.empty():
    print(sample_queue.get())