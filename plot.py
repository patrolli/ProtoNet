import matplotlib.pyplot as plt
import numpy as np

log_path = './logs/2020-03-24_10-05-37/log.txt'
all_loss = []
for i, line in enumerate(open(log_path).readlines()):
    if i == 0:
        continue
    loss = float(line.split(':')[-1])
    all_loss.append(loss)

fig, ax = plt.subplots()
ax.plot(np.linspace(1, 10000, len(all_loss)), all_loss)
ax.set(xlabel='episodes', ylabel='loss', title='miniimagenet,5-way,5-shot')
fig.savefig("loss_of_miniimagenet_5-way_5-shot.png")
fig.show()
