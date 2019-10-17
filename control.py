import sys
import time

import pygame

import numpy as np

import torch
from torch.nn import MSELoss

from lcp_physics.physics.world import World, run_world
from lcp_physics.physics.bodies import Circle, Rect
from lcp_physics.physics.forces import ExternalForce, Gravity
from lcp_physics.physics.constraints import Joint, TotalConstraint
from lcp_physics.physics.utils import Recorder, plot, Defaults

from matplotlib.pyplot import plot, draw, show

DT = Defaults.DT
DTYPE = Defaults.DTYPE

STOP_DIFF = 1e-3
MASS_EPS = 1e-7

runtime = 10
nsteps = 150




def main(screen):
    forces = []
    #ground_truth_mass = torch.tensor([TOTAL_MASS], dtype=DTYPE)

    ct = torch.linspace(0, runtime, nsteps)
    #ut = torch.randn([1,nsteps]).squeeze(0)

    ut = torch.Tensor([-124.5089, -114.6881, -107.8049, -96.0581, -84.1959, -67.6522,
            -48.0741, -24.8150, 2.8534, 32.7275, 62.6945, 86.2345,
            110.7100, 130.5876, 143.7698, 155.8819, 165.4628, 174.4192,
            177.7279, 180.2041, 182.5195, 186.2484, 188.4550, 189.6284,
            186.4164, 53.8088, -174.4299, -183.5373, -179.4377, -180.3835,
            -176.6764, -179.6389, -176.0699, -175.7458, -177.6105, -173.4036,
            -173.7934, -172.6594, -176.7870, -177.1830, -174.3518, -177.3258,
            -172.4485, -172.6221, -174.7140, -176.9489, -174.3911, -175.7147,
            -174.2562, -175.1786, -171.9940, -173.4082, -175.6440, -178.4764,
            -181.5854, -175.9718, -180.1217, -180.0849, -179.2285, -179.9691,
            -178.2708, -180.6049, -180.5375, -178.1284, -181.7236, -183.2817,
            -182.8911, -193.2361, -195.6128, -197.7275, -194.6451, -192.9184,
            -194.7851, -190.7589, -184.8874, -185.4704, -185.3693, -177.2408,
            -176.6083, -149.9813, -159.2283, -143.6585, -135.8553, -133.6324,
            -129.2765, -124.2765, -128.5453, -126.2177, -119.9912, -117.7339,
            -112.9609, -108.5498, -106.2749, -101.7414, -89.0223, -98.5878,
            -101.0174, -100.6900, -100.1747, -103.7277, -97.2610, -98.9044,
            -92.4657, -80.8556, -72.3655, -93.0464, -78.4735, -71.2295,
            -70.5703, -61.7161, -68.1966, -76.4329, -62.1098, -68.6403,
            -75.8038, -73.2225, -48.7771, -41.6379, -48.7434, -55.3279,
            -52.1072, -51.0750, -60.2008, -48.4225, -68.5821, -66.5538,
            -66.0915, -69.5732, -78.5038, -67.5578, -68.5383, -91.4646,
            -65.5343, -57.6426, -74.1637, -69.6933, -91.9495, -109.8934,
            -95.1881, -83.3115, -82.3295, -76.4099, -94.1542, -94.2271,
            -72.7767, -83.3744, -71.8018, -70.7023, -82.7487, -11.7984])

    rec = None
    # rec = Recorder(DT, screen)

    #plot(zhist)

    learning_rate = 0.5
    max_iter = 100


    utT = torch.tensor(ut,requires_grad=True, dtype=DTYPE)
    ctT = torch.tensor(ct,requires_grad=True, dtype=DTYPE)




    optim = torch.optim.RMSprop([utT], lr=learning_rate)

    last_loss = 1e10
    lossHist = []

    for i in range(1,200):
        world, chain = make_world(forces, ctT, utT)

        zhist = positions_run_world(world, run_time=runtime, screen=screen, recorder=rec)
        zhist = torch.stack(zhist)
        optim.zero_grad()

        loss = MSELoss()(zhist, 150*torch.tensor(np.ones(zhist.size()),requires_grad=True, dtype=DTYPE))
        loss.backward()

        lossHist.append(loss.item())
        optim.step()

        print('Loss:', loss.item())
        print('Gradient:', utT.grad)
        print('Next u:', utT)

        plot(lossHist)
        plot(zhist.clone().detach().numpy())

        draw()



    #ground_truth_pos = [p.data for p in ground_truth_pos]
    #ground_truth_pos = torch.cat(ground_truth_pos)

    # learning_rate = 0.5
    # max_iter = 100

    # next_mass = torch.rand_like(ground_truth_mass, requires_grad=True)
    # print('\rInitial mass:', next_mass.item())
    # print('-----')

    # optim = torch.optim.RMSprop([next_mass], lr=learning_rate)
    # loss_hist = []
    # mass_hist = [next_mass.item()]
    # last_loss = 1e10
    # for i in range(max_iter):
    #     #if i % 1 == 0:
    #     #    world, chain = make_world(forces, next_mass.clone().detach(), num_links=NUM_LINKS)
    #     #    run_world(world, run_time=10, print_time=False, screen=None, recorder=None)

    #     world, chain = make_world(forces, next_mass, num_links=NUM_LINKS)
    #     positions = positions_run_world(world, run_time=10, screen=None)
    #     positions = torch.cat(positions)
    #     positions = positions[:len(ground_truth_pos)]
    #     clipped_ground_truth_pos = ground_truth_pos[:len(positions)]

    #     optim.zero_grad()
    #     loss = MSELoss()(positions, clipped_ground_truth_pos)
    #     loss.backward()

    #     optim.step()

    #     print('Iteration: {} / {}'.format(i+1, max_iter))
    #     print('Loss:', loss.item())
    #     print('Gradient:', next_mass.grad.item())
    #     print('Next mass:', next_mass.item())
    #     print('-----')
    #     if abs((last_loss - loss).item()) < STOP_DIFF:
    #         print('Loss changed by less than {} between iterations, stopping training.'
    #               .format(STOP_DIFF))
    #         break
    #     last_loss = loss
    #     loss_hist.append(loss.item())
    #     mass_hist.append(next_mass.item())

    # world = make_world(forces, next_mass)[0]
    # rec = None
    # positions = positions_run_world(world, run_time=30, screen=screen, recorder=rec)
    # positions = torch.cat(positions)
    # positions = positions[:len(ground_truth_pos)]
    # clipped_ground_truth_pos = ground_truth_pos[:len(positions)]
    # loss = MSELoss()(positions, clipped_ground_truth_pos)
    # print('Final loss:', loss.item())
    # print('Final mass:', next_mass.item())

    # plot(loss_hist)
    # plot(mass_hist)



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
