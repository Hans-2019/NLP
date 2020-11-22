import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0,6,0.03)
y = np.sin(x)
plt.plot(x, y, '-')
plt.title('sentence')
plt.savefig("G:\\Code\\NPLhtml\\mysite\\Sta\\img\\result.png")
