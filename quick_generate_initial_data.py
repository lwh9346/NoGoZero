import os
import time
import platform
assert platform.system() == "Windows"
for i in range(0, 16):
    os.system("start cmd /k python data_generator.py random 5 2.0 {}".format(i))
    time.sleep(1)
