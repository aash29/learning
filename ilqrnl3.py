	
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



def f(x):
	theta_dot = x[3]
	
	costheta = torch.cos(x[2])
	sintheta = torch.sin(x[2])
	temp = (polemass_length * x[3] * x[3] * sintheta) / total_mass
	thetaacc = (gravity * sintheta - costheta * temp) / (
	            length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass))
	xacc = temp - polemass_length * thetaacc * costheta / total_mass

	f1 = torch.stack((x[1], xacc, x[3],thetaacc))
	return f1

def g(x):
	theta_dot = x[3]
	
	costheta = torch.cos(x[2])
	sintheta = torch.sin(x[2])
	g1 = torch.tensor([[0],
	[1/total_mass],
	[0],
	[-costheta/(length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass))]])
	return g1



def rhs(x,u):
	return f(x) +g(x)*u


def df(x):
	f1 = f(x)
	j1 = [torch.autograd.grad(f1[i],x,retain_graph=True)[0] for i in range(0,4)]
	#print(j1)
	return torch.transpose(torch.cat(j1,1),0,1)





def iLQR(xinit, uinit, f, g, nhor):
	

	x0 = torch.t(torch.tensor([xinit], dtype=torch.float,requires_grad=True))
	x00 = torch.cat([x0,torch.tensor([[0.0]])],0)
	print(x0)
	u0 = uinit
	f0 = f(x0)
	g0 = g(x0)

	A = df(x0)
	A = torch.cat([A,f0],1)
	A = A*dt

	I1 = torch.tensor([
		[1.0, 0.0, 0.0, 0.0, 0.0],
		[0.0, 1.0, 0.0, 0.0, 0.0],
		[0.0, 0.0, 1.0, 0.0, 0.0],
		[0.0, 0.0, 0.0, 1.0, 0.0]])
	A = A + I1

	A = torch.cat([A, torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]])], 0)
	B = g0*dt
	B = torch.cat([B, torch.tensor([[0.0]])], 0)



	Q = 2*torch.eye(5)

	# #R = torch.tensor([1], dtype=torch.float,requires_grad=True)
	R = 0.4

	N = nhor

	P=[Q]
	K=[torch.tensor([1, 0], dtype=torch.float,requires_grad=True)]


	for n in range(1,N):
		K.append(-(R+torch.t(B).matmul(P[n-1]).matmul(B)).inverse().matmul((torch.t(B)).matmul(P[n-1]).matmul(A)))
		P.append(Q + R*torch.t(K[n]).matmul(K[n])+ torch.t(A+B.matmul(K[n])).matmul(P[n-1]).matmul(A+B.matmul(K[n])))

	# 	#print(n,K[n])


	e = [torch.tensor([[-0.4], [-0.1], [0.1], [-0.1], [0.0]], dtype=torch.float,requires_grad=True)]
	t = [0.0]
	v = [None]*N
	xp =[x00]
	for i in range(1,N):

		v[i-1] = K[N-i].matmul(e[i-1])
		#x.append(A.matmul(x[i-1])+B*u[i-1])
		x1 = torch.cat([x0,torch.tensor([[1.0]])],0) + e[i-1]
		u1 = u0 + v[i-1]
		rhs1 = torch.cat([f(x1)+g(x1)*u1,torch.tensor([[0.0]])],0)


		e.append(x1 + rhs1*dt - x00)
		xp.append(e[i]+x00)
		t.append(i*dt)

	e1 = torch.cat(e,1) 	
	e1 = e1.detach()

	xp = torch.cat(xp,1) 	
	xp = xp.detach()

	return t, xp

[t,x1] = iLQR([1,0,0,0], 0, f, g, 1000)

#plt.plot(x1[0,:].numpy(), x1[2,:].numpy())
plt.plot(t, x1[0,:].numpy())
plt.plot(t, x1[1,:].numpy())
plt.plot(t, x1[2,:].numpy())
plt.plot(t, x1[3,:].numpy())


plt.show()