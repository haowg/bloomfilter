import random
import sys
from math import log, ceil
import numpy as np

class bloomfilter:

    def _random_hash_function():
        # hash function is ax^2 + bx, i.e. pick a and b
        # uniformly at random
        begin = -(1<<31)
        end = (1<<31)
        return (random.randint(begin, end), random.randint(begin, end))


    def __init__(self, memory_usage = 1024, hash_functions_count = 4):
        self._data = [False]*memory_usage
        self._hash_functions = [bloomfilter._random_hash_function()
                               for x in range(hash_functions_count)]


    def __len__(self):
        return len(self._data)


    def __contains__(self, key):
        xs = [(a*key*key + b * key) % len(self)
              for (a,b) in self._hash_functions]
        ys = [self._data[x] for x in xs]
        if ys == [True]*len(self._hash_functions):
            return True
        return False


    def batch_member(self, keys):
        return np.array([k in self for k in keys])

    def batch_insert(self, keys):
        for k in keys:
            self.insert(k)

    def insert(self, key):
        xs = [(a*key*key + b * key) % len(self)
              for (a,b) in self._hash_functions]
        for x in xs:
            self._data[x] = True
        return


class bloomfilter_numpy:

    def __init__(self, memory_usage = 1024, hash_functions_count = 4):
        self._data = np.zeros((memory_usage,), dtype=bool)

        lo = -(1<<31)
        hi = (1<<31)
        self._hash_functions = np.random.randint(lo, hi,
                                                 (hash_functions_count, 2),
                                                 dtype=np.int64)
        self._hash_functions[:, 0] |= 1

    def __len__(self):
        return len(self._data)


    def _santas_little_helper(self, key):
        keys = np.array([key])
        return self.compute_hashes(keys)[:, 0]

    def __contains__(self, key):
        xs = self._santas_little_helper(key)
        ys = self._data[xs]
        return np.all(ys)

    def insert(self, key):
        xs = self._santas_little_helper(key)
        self._data[xs] = True

    def compute_hashes(self, keys):
        a,b = self._hash_functions.T
        a = a.reshape(-1, 1)
        b = b.reshape(-1, 1)
        keys = keys.reshape(1, -1)
        squares = a * (keys * keys)
        linears = b * keys
        mp = 2**31 - 1
        xs = (squares + linears) % mp % (len(self))
        return xs

    def batch_insert(self, keys):
        xs = self.compute_hashes(keys)
        self._data[xs.reshape(-1)] = True


    def _mini_batch(self, keys):
        xs = self.compute_hashes(keys)
        # each column needs to be looked up
        lookups = (self._data[xs.reshape(-1)]).reshape(xs.shape)
        return np.all(lookups, axis=0)


    def batch_member(self, keys):
        batch_size = 2**10
        return np.concatenate([self._mini_batch(keys[i:i+batch_size])
                               for i in range(0, len(keys), batch_size)])

def main():
    # m is bits per element
    n, m = eval(sys.argv[1]), eval(sys.argv[2])
    k = m * log(2)
    #b1 = bloomfilter_numpy(m * n, ceil(k))
    b1 = bloomfilter_numpy(m * n, ceil(k))
    print("Using %d hash functions" % ceil(k))
    b1.batch_insert(10*np.arange(n))

    count_in = 0
    count_out = 0
    queries = 2**26
    elements = np.arange(queries)
    answer = b1.batch_member(elements)
    count_in = answer.sum()
    count_out = len(elements) - count_in
    print("total\treported_in\treported_out\tactual_in")
    print("%d\t%d\t%d\t%d\t%.2f%%" %
          (count_in + count_out, count_in, count_out, n,
           ((count_in - n) / queries)*100))

if __name__ == "__main__":
    main()
