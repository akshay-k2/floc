import numpy as np
import time

from dask.distributed import Client
import dask.array as da

#client = Client("tcp://192.168.1.100:8786")
client = Client("tcp://192.168.0.158:8786")

# Sample computation
x = da.random.random((10000,10000), chunks=(1000,1000))
y = da.exp(x).sum()

print("Number of Pynq Boards in the Cluster: " + str(len(client.scheduler_info()["workers"])))

print("\nStart distributed computation of exponential sum of an array with 100 million random elements...")

start = time.time();
y.compute();
end = time.time()

print("----------------------------------------------")
print("Execution time: " + str(end-start) + " seconds")
print("----------------------------------------------")
