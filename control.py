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

DT = Defaults.DT
DTYPE = Defaults.DTYPE

STOP_DIFF = 1e-3
MASS_EPS = 1e-7

runtime = 10
nsteps = 150




def main(screen):

    #if torch.cuda.is_available():
    #    dev = "cuda:0"
    #else:
    #    dev = "cpu"

    forces = []
    #ground_truth_mass = torch.tensor([TOTAL_MASS], dtype=DTYPE)

    ct = torch.linspace(0, runtime, nsteps, device=Defaults.DEVICE)


    ut = torch.tensor([-1605.5942, -1598.4509, -1554.4310, -1338.5729, -1362.6662, -1601.0057,
        -1368.2047, -1447.8799, -1468.1946, -1561.3396, -1595.0144, -1137.6313,
         -591.9468,  -211.9797,   163.8500,   196.0043,   229.8317,   208.1212,
          234.3518,   111.0826,   103.0544,    58.8933,    27.4864,     7.0699,
         -120.6865,  -240.8309,  -440.6083,  -414.6174,  -418.0714,  -421.2589,
         -352.9074,  -283.0702,  -321.3500,  -262.6164,  -237.7303,  -342.4958,
         -356.1218,  -255.1522,  -372.5189,  -399.5789,  -522.1155,  -460.8092,
         -506.4205,  -380.1456,  -248.6148,  -302.2825,  -103.0938,  -191.7826,
          -61.8832,   233.6888,   731.5193,   832.3451,   842.7044,   976.3949,
          900.6689,   740.5296,   744.6081,   753.7558,   786.8101,   616.9571,
          656.3346,   833.8041,   706.6461,   871.3550,  1030.6888,   900.1758,
          996.5229,   900.0365,   943.0948,   845.3470,   936.0502,   938.4112,
          934.9106,   879.7476,  1083.4686,  1068.4445,  1027.7629,  1061.1221,
         1076.0351,   918.4535,  1024.3541,   989.6422,   923.6206,  1004.4500,
         1041.8034,   838.4601,   803.5565,   772.2364,   636.6075,   936.8204,
          675.2448,   678.9081,  1004.8696,   979.5466,   652.9930,   820.7622,
          777.7355,   728.2931,   876.9120,   810.3975,   855.0797,   769.3505,
          701.5111,   668.2611,   744.0883,   456.1483,   536.5857,   540.6397,
          439.4413,   524.1302,   233.6891,   506.7044,   365.0381,   185.3458,
          231.1983,   347.6416,   356.6151,   279.0252,   325.3887,   415.5265,
          354.4233,   270.3732,   231.2778,   216.2692,   345.1276,   185.7488,
          213.7231,   331.3090,  -108.3021,   298.4766,    51.4663,    -6.6531,
           93.6787,    70.1087,   135.1969,   -57.0313,    10.2455,    34.4662,
          -79.4893,   -58.0966,  -121.3225,  -228.2877,  -125.6331,  -153.7666,
           -9.6513,  -135.1915,   -94.7192,   -87.6338,  -168.4967,   155.2225],
             device=Defaults.DEVICE)

    #ut = 100*torch.randn([1,nsteps]).squeeze(0)

    rec = None
    # rec = Recorder(DT, screen)

    #plot(zhist)

    learning_rate = 0.5
    max_iter = 100


    utT = torch.tensor(ut,requires_grad=True, dtype=DTYPE,device=Defaults.DEVICE)
    ctT = torch.tensor(ct,requires_grad=True, dtype=DTYPE,device=Defaults.DEVICE)




    optim = torch.optim.RMSprop([utT], lr=learning_rate)

    last_loss = 1e10
    lossHist = []

    for i in range(1,2000):
        world, chain = make_world(forces, ctT, utT)

        zhist = positions_run_world(world, run_time=runtime, screen=None, recorder=rec)
        zhist = torch.stack(zhist).to(device=Defaults.DEVICE)
        optim.zero_grad()

        loss = MSELoss()(zhist, 150*torch.tensor(np.ones(zhist.size()),requires_grad=True, dtype=DTYPE,device=Defaults.DEVICE))/100
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

        plt.plot(lossHist)
        plt.draw()

        pl1 = zhist.cpu()
        plt.plot(pl1.clone().detach().numpy())
        #plt.plot(zhist)
        plt.draw()

        plt.pause(0.001)

        utTCurrent = utT.clone()

    world, chain = make_world(forces, ctT, utTCurrent)

    positions_run_world(world, run_time=runtime, screen=None, recorder=rec)





def make_world(forces, controlT, controlU):

    def controlForce(t):
        #cu = controlU.clone().detach()
        #mag = torch.tensor(np.interp(t, controlT, controlU), requires_grad=True, dtype=DTYPE )
        #mag = controlU[0]+controlU[1]
        t1 = torch.Tensor([t])

        #mag = None
        #mag = Interp1d()(controlT, controlU, t1, mag)

        dt = controlT[1]-controlT[0]
        i = int(t // dt.item())

        #interp = Interpolate(controlT,controlU)
        mag = controlU[i]

        return -mag*ExternalForce.ROT

    bodies = []
    joints = []

    # make chain of rectangles

    r = Rect([0, 50, 240], [60, 60], mass = 1)
    #r.set_p(r.p.new_tensor([1, 1, 1]))
    bodies.append(r)
    #joints.append(Joint(r, None, [300, 30]))
    r.add_force(Gravity(g=10))


    cf = ExternalForce(controlForce,multiplier=1)

    r.add_force(cf)


    floor = Rect([0, 300], [1000, 30], mass=100)
    joints.append(TotalConstraint(floor))
    bodies.append(floor)
        #joints.append(Joint(bodies[-1], bodies[-2], [300, 25 + 50 * i]))
        #bodies[-1].add_no_contact(bodies[-2])

    # make projectile
    #m = 3
    #c_pos = torch.tensor([50, bodies[-1].pos[1]])  # same Y as last chain link
    #c = Circle(c_pos, 20, restitution=1.)
    #bodies.append(c)
    #for f in forces:
    #    c.add_force(ExternalForce(f, multiplier=500 * m))

    world = World(bodies, joints, dt=DT, post_stab=True)
    return world, r


def positions_run_world(world, dt=Defaults.DT, run_time=10,
                        screen=None, recorder=None):
    positions = [world.bodies[0].p[1]]
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

        positions.append(world.bodies[0].p[1])
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
    return positions


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
