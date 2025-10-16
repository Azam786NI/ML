import numpy as np


def relu(lis):
    new = []
    for i in lis[0]:
        if i > 0:
            new.append(1)
        else:
            new.append(0)

    return np.array([new])


loss = np.array([[0.61237244, 0.73484692]])

a2 = np.array([[0, 0, 1.22474487]])

print("---- gradient w.r.t wo", a2.T @ loss)
print("---- gradient w.r.t bo", loss)

print("\n\n")

w3 = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
z_norm2 = np.array([[-1.2247448713915885, 0.0, 1.2247448713915896]])


g = loss @ w3.T
g *= relu(a2)


print("---- gradient w.r.t alpha2", g * z_norm2)
print("---- gradient w.r.t beta2", g)
