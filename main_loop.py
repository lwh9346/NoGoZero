import torch
from train import train
from multiprocessing import Process
import os
import time

if __name__ == "__main__":
    train(epoches=20)
    i = 1
    while True:
        tasks = []
        for j in range(4):
            p = Process(target=os.system, args=[
                        "python data_generator.py model_{} 20 2.0 {}".format(i, j)])
            p.start()
            tasks.append(p)
            time.sleep(0.1)
        for t in tasks:
            t.join()
        train(model="model_{}".format(i),
              save_name="model_{}".format(i+1),
              previous="model_{}".format(i),
              epoches=5)
        i += 1
