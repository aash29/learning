	
import torch
import matplotlib.pyplot as plt
from torch.nn import MSELoss

dt=0.01


gravity = 9.8
masscart = 1.0
masspole = 0.1
total_mass = (masspole + masscart)
length = 1 # actually half the pole's length
polemass_length = (masspole * length)
force_mag = 10.0

force = 0

x = torch.tensor([[0.0], [0.0], [0.0], [0.0]], dtype=torch.float,requires_grad=True)


theta_dot = x[3]

costheta = torch.cos(x[2])
sintheta = torch.sin(x[2])
temp = (polemass_length * x[3] * x[3] * sintheta) / total_mass
thetaacc = (gravity * sintheta - costheta * temp) / (
            length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass))
xacc = temp - polemass_length * thetaacc * costheta / total_mass



f1 = torch.stack((x[1], xacc, x[3],thetaacc))

g1 = torch.tensor([[0],
	[1/total_mass],
	[0],
	[-costheta/(length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass))]])



j1 = [torch.autograd.grad(f1[i],x,retain_graph=True)[0] for i in range(0,4)]
torch.cat(j1,1)



#print(j1)


A = torch.cat(j1,1)
A = A*dt+torch.eye(4)

B = g1*dt


# B = torch.tensor([[1], [0]], dtype=torch.float,requires_grad=True)
# B = B*dt
# Q = torch.tensor([[2, 0], [0, 1]], dtype=torch.float,requires_grad=True)

# #R = torch.tensor([1], dtype=torch.float,requires_grad=True)
# R = 1.4

# N = 500

# P=[Q]
# K=[torch.tensor([1, 0], dtype=torch.float,requires_grad=True)]
# u = [None]*N

# for n in range(1,N):
# 	K.append(-(R+torch.t(B).matmul(P[n-1]).matmul(B)).inverse().matmul((torch.t(B)).matmul(P[n-1]).matmul(A)))
# 	P.append(Q + R*torch.t(K[n]).matmul(K[n])+ torch.t(A+B.matmul(K[n])).matmul(P[n-1]).matmul(A+B.matmul(K[n])))

# 	#print(n,K[n])


# x = [torch.tensor([[5], [-7]], dtype=torch.float,requires_grad=True)]

# for i in range(1,N):
# 	u[i-1] = K[N-i].matmul(x[i-1])
# 	x.append(A.matmul(x[i-1])+B*u[i-1])

# x1 = torch.cat(x,1) 	

# x1 = x1.detach()

# plt.plot(x1[0,:].numpy(), x1[1,:].numpy())
# plt.show()