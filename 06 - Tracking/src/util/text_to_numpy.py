import numpy as np


data = []

with open("./data/video1_truth.txt") as f:
    lines = f.readlines()

    for line in lines:
        x, y = line.split(", ")
        data.append((int(x), int(y)))

# print(data)
print(len(data))

np.save("./data/video1_truth", np.array(data))
