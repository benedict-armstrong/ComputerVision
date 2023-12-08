import numpy as np


data = []

with open("out/video3.txt") as f:
    lines = f.readlines()

    for line in lines:
        x, y = line.split(", ")
        data.append((int(x), int(y)))

print(data)

np.save("out/video3_truth", np.array(data))
