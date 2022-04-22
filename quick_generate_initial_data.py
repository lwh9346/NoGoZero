import os
import time
from multiprocessing import Process
if __name__ == "__main__":
    j = 0
    NUM_WORKERS = 16
    while True:
        tasks = []
        for i in range(j, j+NUM_WORKERS):
            p = Process(target=os.system, args=[
                "python3 data_generator.py random 20 2.0 {}".format(i)])
            tasks.append(p)
            p.start()
            time.sleep(0.1)
        for p in tasks:
            p.join()
        j += NUM_WORKERS
