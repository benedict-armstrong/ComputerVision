import numpy as np


data = []

with open("./data/video3_truth.txt") as f:
    lines = f.readlines()

    for line in lines:
        x, y = line.split(", ")
        data.append((int(x), int(y)))

print(data)

np.save("./data/video3_truth", np.array(data))
