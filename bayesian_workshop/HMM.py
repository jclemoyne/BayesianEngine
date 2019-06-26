import time
# non-cached recursion

fib_cache = dict()


def fib(x):

    if x == 0:
        return 0
    elif x == 1:
        return 1
    else:
        return fib(x-1) + fib(x-2)


def fib2(x):

    if fib_cache.get(x) is not None:
        return fib_cache[x]

    result = None
    if x == 0:
        result = 0
    elif x == 1:
        result = 1
    else:
        result = fib(x-1) + fib(x-2)

    if fib_cache.get(x) is  None:
        fib_cache[x] = result

    return result


def run_fib():
    startTime = time.time()
    print("%-14s:%d" % ("Result", fib(32)))
    print("%-14s:%.4f seconds" % ("Elapsed time", time.time() - startTime))


def run_fib2():
    startTime = time.time()
    print("%-14s:%d" % ("Result", fib2(32)))
    print("%-14s:%.4f seconds" % ("Elapsed time", time.time() - startTime))


if __name__ == '__main__':
    run_fib()
    run_fib2()
