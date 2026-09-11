import numpy as np
import room
from mpi4py import MPI
import time

from tags import *
import plotting

# TODO: move to sparse matrices. <- for A, b in the room's eq.sys
# TODO: update plotting to merge u matrices first.

# parameters
gridsize = 100
assert gridsize > 3  # must be > 3
h = 1 / gridsize
# condition size
c_size = gridsize - 1
c_size_small = c_size//2
# fixed wall temperatures:
T = {"w": 15,
     "h": 40,
     "wf": 5}
iterations = 10
w = 0.8

comm = MPI.Comm.Clone(MPI.COMM_WORLD)
# process 0 coordinates, summarizes data and plots.
# processes 1,2,3,4 have their own room

if comm.Get_size() < 5:
    if comm.get_rank() > 0:
        exit()
    raise Exception(f"Too few processes ({comm.Get_size()}), 5 is ideal.")

rank = comm.Get_rank()

if rank == 0:
    # coordination
    # the first two rows of the coordinating conditions use the full row extent.
    # the third row only uses half
    dirs = T['w'] * np.ones((3,c_size), dtype='d')
    neus = np.zeros((3,c_size), dtype='d')
    dummy = np.empty(1, dtype='d')
    u = [None] * 4

    t = time.time()

    for k in range(iterations):
        # Solve room 2:
        t_old, t = t, time.time()
        print(f"starting iteration {k}, time since last: {t-t_old}")
        with open("output.log", mode='w') as f:
            f.write(f"starting iteration {k}, time since last: {t-t_old}")
        comm.Send([dirs, MPI.DOUBLE], dest=2, tag=TAG_TO_ROOM)
        comm.Recv([neus, MPI.DOUBLE], source=2, tag=TAG_FROM_ROOM)

        # Solve room 1 and 3:
        comm.Isend([neus[0], MPI.DOUBLE], dest=1, tag=TAG_TO_ROOM)
        comm.Isend([neus[1], MPI.DOUBLE], dest=3, tag=TAG_TO_ROOM)
        comm.Isend([neus[2,:c_size_small], MPI.DOUBLE], dest=4, tag=TAG_TO_ROOM)

        comm.Irecv([dirs[0], MPI.DOUBLE], source=1, tag=TAG_FROM_ROOM)
        comm.Irecv([dirs[1], MPI.DOUBLE], source=3, tag=TAG_FROM_ROOM)
        comm.Recv([dirs[2,:c_size_small], MPI.DOUBLE], source=4, tag=TAG_FROM_ROOM)

        if k == iterations-1:
            for r in range(1,5):
                comm.Isend([dummy, MPI.DOUBLE], dest=r, tag=TAG_SEND_U)

            for r in range(1,5):
                u[r-1] = comm.recv(source=r, tag=TAG_U_FROM_ROOM)

            plotting.plot_u_ext(u, show_labels=False, timeout=None)


    # Send shutdown commands
    for r in range(1,5):
        comm.Send([dummy, MPI.DOUBLE], dest=r, tag=TAG_STOP)


elif rank == 1:
    r1 = room.Room1(comm, gridsize, T)
    r1.run(w)

elif rank == 2:
    r2 = room.Room2Ext(comm, gridsize, T)
    r2.run(w)

elif rank == 3:
    r3 = room.Room3(comm, gridsize, T)
    r3.run(w)

elif rank == 4:
    r4 = room.Room4(comm, gridsize, T)
    r4.run(w)

print(f"Process {rank} finished")
