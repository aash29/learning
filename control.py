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


TIME = 40
DT = Defaults.DT
DTYPE = Defaults.DTYPE

STOP_DIFF = 1e-3
MASS_EPS = 1e-7

runtime=20



def main(screen):
    forces = []
    #ground_truth_mass = torch.tensor([TOTAL_MASS], dtype=DTYPE)

    ct = np.linspace(0, runtime,20)
    ut = np.ones([1,605]).squeeze(0)

    rec = None
    # rec = Recorder(DT, screen)

    #plot(zhist)

    learning_rate = 0.5
    max_iter = 100


    #utT = torch.tensor(ut,requires_grad=True)

    ground_truth_mass = torch.tensor([1.0], dtype=DTYPE)


    utest = torch.rand_like(ground_truth_mass, requires_grad=True, dtype=DTYPE)

    optim = torch.optim.RMSprop([utest], lr=learning_rate)

    last_loss = 1e10

    world, chain = make_world(forces, ct, utest)

    zhist = positions_run_world(world, run_time=runtime, screen=screen, recorder=rec)
    zhist = torch.cat(zhist)
    optim.zero_grad()

    loss = MSELoss()(zhist, torch.tensor(np.zeros(zhist.size()),requires_grad=True))
    loss.backward()
    optim.step()

    print('Loss:', loss.item())
    print('Gradient:', utest.grad.item())
    print('Next mass:', utest.item())




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
        #mag = np.interp(t, controlT, controlU.detach().numpy())
        mag = 1
        return -mag*ExternalForce.DOWN

    bodies = []
    joints = []

    # make chain of rectangles

    r = Rect([500, 50], [60, 60], mass = controlU)
    r.set_p(r.p.new_tensor([1, 1, 1]))
    bodies.append(r)
    #joints.append(Joint(r, None, [300, 30]))
    r.add_force(Gravity(g=10))


    cf = ExternalForce(controlForce,multiplier=controlU)

    r.add_force(cf)


    floor = Rect([30, 600], [1000, 30], mass=100)
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
    #positions = [world.bodies[0].p[2]]
    positions = [torch.cat([b.p for b in world.bodies])]

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

        positions.append(world.bodies[0].p[2])
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
