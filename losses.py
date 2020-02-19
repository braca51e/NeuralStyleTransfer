import matplotlib.pyplot as plt

# Plotting losses

f = open("images/results/dragonblue/losses.txt", "r")

x = []
y = []

for line in f.readlines():
    tokens = line.split("=")
    loss_value = float(tokens[1].strip())
    y.append(loss_value)

for i in range(0, len(y)):
    x.append(i)

plt.plot(x, y)
plt.show()
