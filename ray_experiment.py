import ray
import time

ray.init()


@ray.remote
def function1(x):
    time.sleep(1)
    return x + 1


@ray.remote
def function2(x):
    time.sleep(2)
    return x - 1


@ray.remote
def function3(x):
    time.sleep(1.5)
    return x * 2


# Launch subprocesses for different functions
futures1 = [function1.remote(i) for i in range(5)]
futures2 = [function2.remote(i) for i in range(5)]
futures3 = [function3.remote(i) for i in range(5)]

# Collect all futures
all_futures = futures1 + futures2 + futures3

# Retrieve the results
results = ray.get(all_futures)
print(results)

ray.shutdown()
