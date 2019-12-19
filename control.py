import sys
import time

import pygame

import numpy as np

import torch
from torch.nn import MSELoss

import sys
sys.path.insert(0, '/home/aash29/cpp/lcp-physics')

from lcp_physics.physics.world import World, run_world
from lcp_physics.physics.bodies import Circle, Rect
from lcp_physics.physics.forces import ExternalForce, Gravity
from lcp_physics.physics.constraints import Joint, TotalConstraint
from lcp_physics.physics.utils import Recorder, plot, Defaults

import matplotlib.pyplot as plt

from datetime import datetime
import os

DT = Defaults.DT
DTYPE = Defaults.DTYPE

STOP_DIFF = 1e-3
MASS_EPS = 1e-7

runtime = 15
nsteps = 200

ct = torch.linspace(0, runtime, nsteps, device=Defaults.DEVICE)

ut = torch.tensor([ 2.1974e+02, -3.9461e+02,  2.2012e+02, -2.9735e+03,  3.2030e+02,
         2.2827e+03,  1.2295e+03, -1.4982e+02, -1.9248e+03,  1.1052e+03,
        -5.3387e+02,  7.8415e+02,  6.3802e+00,  1.1669e+03,  7.0902e+02,
        -3.6570e+02, -3.8780e+02, -6.9890e+02, -8.4595e+02, -1.8133e+02,
        -1.8723e+03,  5.0338e+02,  3.6651e+02, -6.1630e+02, -3.8698e+02,
         6.1566e+02,  1.7732e+03, -6.1013e+02, -1.2687e+03,  1.5174e+03,
        -2.4781e+02,  5.5225e+02,  1.6877e+03,  3.9037e+02, -2.4712e+02,
         1.4431e+03,  7.7652e+02, -4.4228e+02, -1.2974e+03, -6.9668e+01,
        -1.0253e+03, -1.0163e+03,  1.0441e+02,  7.3002e+01,  7.9560e+02,
         7.9767e+02, -5.5569e+01,  1.0546e+03, -8.0802e+02, -1.4638e+03,
        -1.4052e+03,  1.0045e+03, -1.4037e+02, -1.1344e+03,  6.7412e+02,
         5.4585e+02, -1.9265e+03,  1.3628e+02, -4.5915e+02, -2.3004e+03,
         8.8670e+02,  1.0187e+03,  4.8258e+02, -5.5115e+02,  1.2971e+03,
        -1.7697e+03, -9.5966e+02, -4.1746e+02,  1.7301e+03, -1.6286e+01,
         1.6780e+02,  3.9109e+02,  1.5577e+03,  1.7987e+03, -2.2594e+03,
         1.9357e+03, -6.3047e+02, -2.8470e+02, -1.9689e+02, -1.4915e+03,
         1.3701e+03, -4.4875e+02,  1.5385e+03, -7.4316e+02,  5.2380e+02,
        -8.2932e+02, -1.0557e+03, -1.8971e+03, -1.6471e+02,  4.3995e+02,
         1.6862e+03,  6.2494e+02,  5.0747e+02,  7.7213e+02, -2.6381e+02,
        -8.4521e+02,  6.1137e+01,  4.7627e+02,  9.4569e+02,  4.4757e+02,
        -1.2819e+03,  1.0399e+03,  2.6586e+03,  1.6766e+02, -9.9907e+02,
         1.2759e+03, -1.4931e+03,  1.7179e+02, -4.3393e+02,  8.5506e+01,
         3.8040e+00, -8.7190e+02,  2.4780e+03,  1.6320e+02, -1.3740e+03,
         1.0321e+03, -4.6568e+02,  2.2199e+02, -3.5904e+02,  9.4419e+02,
         1.6735e+03, -3.8307e+02, -7.9282e+02,  5.8832e+02,  3.2352e+02,
        -1.6934e+03, -2.6222e+02,  9.1440e+02,  2.2734e+02,  1.7867e+03,
         2.3776e+02,  4.7743e+02,  9.8050e+02, -7.2339e+02,  1.3852e+03,
         5.3358e+02, -2.5250e+02, -8.4140e+01, -1.3779e+03, -1.0969e+03,
         5.3103e+02, -6.3520e+02,  7.8873e+02, -1.6889e+02,  1.5488e+03,
         1.7601e+03,  3.8349e+02, -9.1108e+02,  7.8040e+02, -4.0401e+02,
        -1.7702e+03,  1.9424e+03, -5.8680e+02,  3.0000e+02,  1.3890e+03,
        -2.4012e+03,  1.1461e+03,  1.7252e+02,  1.9408e+03,  1.8831e+02,
         7.4990e+02,  4.5020e+02,  1.0851e+02, -2.1388e+02, -3.9930e+02,
         1.4384e+03, -9.9174e+02, -8.5502e+02,  1.5120e+03,  1.0964e+03,
         7.6674e+01, -1.1191e+03,  6.2575e+02,  5.9917e+02,  1.8816e+03,
         1.9389e+03,  2.5992e+02, -8.6894e+02, -8.1469e+02, -3.3213e+02,
         1.2820e+02,  1.7652e+03, -1.4432e+01, -3.2869e+02,  2.5350e+02,
         2.5054e+02, -1.3967e+03, -1.1628e+03,  2.2301e+02,  2.9176e+02,
         5.5634e+02, -1.3254e+03, -8.1058e+02, -1.4025e+03, -6.3080e+02,
         1.4860e+03,  3.5566e+02, -5.4396e+02,  1.2204e+03, -7.7049e+02,
         7.6982e+02,  1.0310e+03,  4.0839e+02, -1.5185e+03,  3.0462e+02,
         6.6783e+02, -9.9945e+02,  1.6060e+02, -2.7067e+02, -1.5099e+03,
         1.0684e+03, -1.1045e+03,  3.0731e+03,  7.4982e+02, -3.4904e+02,
        -2.5883e+02,  5.2936e+00,  4.7425e+02, -9.2592e+02,  6.2346e+02,
        -5.5579e+02, -5.3606e+02, -1.3821e+03,  1.4406e+03, -4.9622e+02,
        -6.3113e+02, -1.4432e+01,  9.4980e+02, -8.4093e+02,  2.2076e+02,
        -1.4980e+03, -7.8047e+02, -7.0492e+01, -2.6725e+03, -1.0978e+03,
        -9.3416e+02,  5.8107e+02, -4.0320e+02, -1.1730e+03,  1.2252e+03,
        -2.4075e+02,  7.9359e+02, -1.2165e+02,  1.5768e+03, -8.6270e+02,
        -1.8978e+03, -1.0989e+03,  1.6260e+03,  6.2305e+02,  3.0064e+03,
        -5.4735e+02, -9.9640e+02, -7.5404e+02,  6.2146e+02,  4.6612e+02,
         8.6640e+01, -5.4836e+01, -8.5650e+01,  2.9837e+02, -1.4595e+03,
        -1.4806e+03, -3.9277e+02, -6.2414e+02,  1.5873e+03, -5.2005e+01,
        -1.1508e+03,  2.1107e+03, -1.0810e+03,  1.0646e+03,  3.1884e+02,
         2.0282e+03, -1.4538e+03, -1.2956e+03, -2.0633e+03,  4.1090e+02,
        -3.6059e+01,  2.4278e+03, -2.0842e+02,  1.4277e+02, -1.1413e+03,
         8.2271e+01,  5.7034e+02, -9.0853e+02,  4.5815e+02,  7.5502e+02,
        -1.2754e+03, -1.7135e+03, -1.0959e+03, -1.7417e+02, -7.9031e+02,
        -2.5303e+02,  8.1210e+02, -2.5364e+03,  1.7334e+02,  1.7588e+03,
        -5.2116e+02,  7.2254e+02,  1.3499e+03, -2.2103e+01, -8.0952e+02,
         3.8531e+02, -1.1212e+02,  3.5829e+02,  1.9986e+02, -9.7434e+02,
         1.5634e+03,  5.0316e+02, -4.7658e+02,  5.3676e+02,  2.5295e+01,
         1.1404e+03,  1.2255e+03,  2.0662e+03, -7.9968e+02, -1.3283e+03,
         1.5854e+03,  1.9172e+03, -8.9376e+02,  3.4392e+01,  1.0265e+01,
         9.7785e+02,  1.7200e+03,  2.6331e+02, -2.3878e+02,  1.8870e+02,
         5.5916e+02, -4.4190e+02, -4.3847e+02,  1.6764e+03, -1.3924e+02,
        -1.4246e+03,  3.0750e+03, -6.5611e+02, -3.0371e+02,  6.4739e+02,
         1.1493e+03,  5.8145e+01, -7.4828e+02, -6.2635e+02,  2.6479e+02,
        -8.9066e+02,  6.9268e+02, -4.1693e+02,  1.6600e+03,  5.2641e+02,
        -9.3616e+02,  5.3181e+02, -3.4430e+02, -6.3761e+02, -3.6419e+02,
         1.5932e+02,  4.7210e+01,  4.9088e+01,  1.5566e+02,  5.8079e+00,
         2.9166e+02,  1.6411e+03,  2.8043e+01,  4.9992e+02,  3.3693e+02,
        -3.6641e+02, -3.9589e+03,  1.4472e+03, -1.6358e+03, -4.1586e+02,
         1.0473e+03, -5.5641e+02, -8.9895e+02, -4.0383e+02, -1.8502e+01,
         1.2166e+03,  6.1787e+02,  8.7153e+01, -7.4040e+02,  1.0871e+01,
        -1.3293e+03,  5.4521e+02,  2.3119e+02, -6.4541e+02,  5.6817e+02,
         1.2659e+03, -2.4496e+03, -7.8727e+02,  8.6269e+02,  1.0020e+03,
         3.7808e+02, -1.0823e+03,  1.1777e+03,  2.7246e+03, -1.9500e+02,
         2.0510e+03,  9.6160e+02,  8.9854e+01, -1.3354e+03, -1.1571e+03,
         2.6926e+02,  1.8035e+02,  1.0049e+03,  4.6728e+02, -7.6358e+02],
                  device=Defaults.DEVICE)

#ut = 1000*torch.randn([1,nsteps*2],device=Defaults.DEVICE).squeeze(0)

utT = torch.tensor(ut, requires_grad=True, dtype=DTYPE, device=Defaults.DEVICE)
ctT = torch.tensor(ct, requires_grad=True, dtype=DTYPE, device=Defaults.DEVICE)





def main(screen):
    dateTimeObj = datetime.now()


    timestampStr = dateTimeObj.strftime("%d-%b-%Y(%H:%M)")
    print('Current Timestamp : ', timestampStr)

    if not os.path.isdir(timestampStr):
        os.mkdir(timestampStr)

    #if torch.cuda.is_available():
    #    dev = "cuda:0"
    #else:
    #    dev = "cpu"

    forces = []






    rec = None
    #rec = Recorder(DT, screen)

    #plot(zhist)

    learning_rate = 0.5
    max_iter = 100

    plt.subplots(21)
    plt.subplot(212)
    plt.gca().invert_yaxis()




    optim = torch.optim.RMSprop([utT], lr=learning_rate)

    last_loss = 1e10
    lossHist = []

    for i in range(1,20000):
        world, chain = make_world(forces, ctT, utT)

        hist = positions_run_world(world, run_time=runtime, screen=screen, recorder=rec)
        xy = hist[0]
        vel = hist[1]
        control = hist[2]
        xy = torch.cat(xy).to(device=Defaults.DEVICE)
        x1 = xy[0::2]
        y1 = xy[1::2]
        control = torch.cat(control).to(device=Defaults.DEVICE)
        optim.zero_grad()

        targetxy = []
        targetControl = []
        j = 0
        while (j<xy.size()[0]):
            targetxy.append(500)
            targetxy.append(240)

            targetControl.append(0)
            targetControl.append(0)

            j = j + 2



        tt = torch.tensor(targetxy, requires_grad=True, dtype=DTYPE,device=Defaults.DEVICE).t()
        tc = torch.tensor(targetControl, requires_grad=True, dtype=DTYPE, device=Defaults.DEVICE).t()
        #loss = MSELoss()(zhist, 150*torch.tensor(np.ones(zhist.size()),requires_grad=True, dtype=DTYPE,device=Defaults.DEVICE))/100
        #loss = MSELoss()(xy, tt) / 10 + 0*MSELoss()(control, tc) / 1000 + abs(vel[-1][0]) + abs(vel[-1][1]) + abs(vel[-1][2])

        loss = MSELoss()(xy, tt) / 10 +  0*MSELoss()(control, tc) / 1000 + abs(vel[-1][0]) + abs(vel[-1][1]) + abs(vel[-1][2])
        #loss = zhist[-1]
        loss.backward()

        lossHist.append(loss.item())
        optim.step()

        print('Loss:', loss.item())
        print('Gradient:', utT.grad)
        print('Next u:', utT)

        #plt.axis([-1, 1, -1, 1])
        plt.ion()
        plt.show()

        plt.subplot(211)

        plt.plot(lossHist)
        plt.draw()

        pl1 = xy.cpu()
        plt.subplot(212)

        plt.plot(pl1.clone().detach().numpy()[::4],pl1.clone().detach().numpy()[1::4])
        #plt.plot(zhist)
        plt.draw()

        plt.pause(0.001)

        plt.savefig('2step.png')


        if (i%20) == 0:
            plt.subplot(211)
            plt.cla()
            plt.subplot(212)
            plt.cla()
            plt.gca().invert_yaxis()
            lossHist.clear()
            torch.save(utT, timestampStr+'/file'+str(i)+'.pt')

        utTCurrent = utT.clone()

    world, chain = make_world(forces, ctT, utTCurrent)

    positions_run_world(world, run_time=runtime, screen=None, recorder=rec)




def controlForce(t, controlT, controlU):

    dt = controlT[1]-controlT[0]
    i = int(t // dt.item())
    mag = controlU[i]

    return -mag*ExternalForce.ROT

def controlForce2(t, controlT, controlU):

    dt = controlT[1]-controlT[0]
    i = int(t // dt.item())

    mag = controlU[nsteps+i]

    return -mag*ExternalForce.ROT



def make_world(forces, controlT, controlU):




    bodies = []
    joints = []

    # make chain of rectangles

    r = Rect([0, 120, 240], [60, 60], mass = 1 )

    r2 = Rect([0, 200, 240], [60, 60], mass = 1)
    #r.set_p(r.p.new_tensor([1, 1, 1]))
    bodies.append(r)
    bodies.append(r2)
    #joints.append(Joint(r, None, [300, 30]))
    r.add_force(Gravity(g=10))
    r2.add_force(Gravity(g=10))

    controlForceL = lambda t: controlForce(t, controlT, controlU)
    cf = ExternalForce(controlForceL,multiplier=1)

    controlForceL2 = lambda t: controlForce2(t, controlT, controlU)
    cf2 = ExternalForce(controlForceL2, multiplier=1)

    r.add_force(cf)
    r2.add_force(cf2)


    floor = Rect([0, 300], [1000, 30], mass=100)
    floorStep = Rect([300, 240], [100, 89], mass=100)
    floorStep2 = Rect([0, 240], [100, 89], mass=100)



    joints.append(TotalConstraint(floor))
    bodies.append(floor)
    joints.append(TotalConstraint(floorStep))
    bodies.append(floorStep)
    joints.append(TotalConstraint(floorStep2))
    bodies.append(floorStep2)

        #joints.append(Joint(bodies[-1], bodies[-2], [300, 25 + 50 * i]))
        #bodies[-1].add_no_contact(bodies[-2])

    # make projectile
    #m = 3
    #c_pos = torch.tensor([50, bodies[-1].pos[1]])  # same Y as last chain link
    #c = Circle(c_pos, 20, restitution=1.)
    #bodies.append(c)
    #for f in forces:
    #    c.add_force(ExternalForce(f, multiplier=500 * m))

    world = World(bodies, joints, dt=DT, post_stab=True,strict_no_penetration=True)
    return world, r


def positions_run_world(world, dt=Defaults.DT, run_time=10,
                        screen=None, recorder=None):
    positions = [world.bodies[0].p[1:]]
    vel = [world.bodies[0].v]
    control = [controlForce(0, ctT, utT)[0].unsqueeze(0), controlForce2(0, ctT, utT)[0].unsqueeze(0)]

    #positions = [torch.cat([b.p for b in world.bodies])]

    if screen is not None:
        background = pygame.Surface(screen.get_size())
        background = background.convert()
        background.fill((255, 255, 255))
        
        

        animation_dt = dt
        elapsed_time = 0.
        prev_frame_time = -animation_dt
        start_time = time.time()


    while world.t < run_time:
        world.step()

        positions.append(world.bodies[0].p[1:])
        control.append(controlForce(world.t, ctT, utT)[0].unsqueeze(0))
        control.append(controlForce2(world.t, ctT, utT)[0].unsqueeze(0))
        vel.append(world.bodies[0].v)
        #positions.append(torch.cat([b.p for b in world.bodies]))
        if screen is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            if elapsed_time - prev_frame_time >= animation_dt or recorder:
                prev_frame_time = elapsed_time

                screen.blit(background, (0, 0))
                pygame.display.flip()
                
                update_list = []
                for body in world.bodies:
                    update_list += body.draw(screen)
                for joint in world.joints:
                    update_list += joint[0].draw(screen)

                if not recorder:
                    # Don't refresh screen if recording
                    #screen.fill((0, 0, 0))
                    pygame.display.update(update_list)
                else:
                    recorder.record(world.t)

            elapsed_time = time.time() - start_time
            # print('\r ', '{} / {}  {} '.format(int(world.t), int(elapsed_time),
            #                                   1 / animation_dt), end='')
    return [positions,vel, control]


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '-nd':
        # Run without displaying
        screen = None
    else:
        width, height = 1000, 600
        screen = pygame.display.set_mode((width, height), pygame.HWSURFACE)
        screen.set_alpha(None)
        pygame.display.set_caption('control')
        print(screen)

    main(screen)
