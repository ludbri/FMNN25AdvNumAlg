import numpy as np
import room
from mpi4py import MPI

from tags import *
import plotting

# TODO: move to sparse matrices? <- for A in the room eq.sys
# TODO: something looks weird. Maybe direction of Neumann conditions?

# parameters
# roomsizes = np.array([[1,1],[2,1],[1,1]])
gridsize = 3
h = 1 / gridsize
# condition size
c_size = gridsize - 1
# fixed wall temperatures:
T = {"w": 15,
     "h": 40,
     "wf": 5}
w = 0.8

comm = MPI.Comm.Clone(MPI.COMM_WORLD)
# process 0 coordinates, summarizes data and plots.
# processes 1,2,3 have their own room

if comm.Get_size() < 4:
    if comm.get_rank() > 0:
        exit()
    raise Exception(f"Too few processes ({comm.Get_size()}), 4 is ideal.")

rank = comm.Get_rank()

if rank == 0:
    # coordination
    dirs = np.zeros((2,c_size), dtype='d')
    neus = np.zeros((2,c_size), dtype='d')
    dummy = np.empty(1, dtype='d')
    u = [None] * 3

    for k in range(5):
        # Solve room 2:
        comm.Send([dirs, MPI.DOUBLE], dest=2, tag=TAG_TO_ROOM)
        comm.Recv([neus, MPI.DOUBLE], source=2, tag=TAG_FROM_ROOM)

        # Solve room 1 and 3:
        comm.Isend([neus[0], MPI.DOUBLE], dest=1, tag=TAG_TO_ROOM)
        comm.Isend([neus[1], MPI.DOUBLE], dest=3, tag=TAG_TO_ROOM)

        comm.Irecv([dirs[0], MPI.DOUBLE], source=1, tag=TAG_FROM_ROOM)
        comm.Recv([dirs[1], MPI.DOUBLE], source=3, tag=TAG_FROM_ROOM)

        print(f"iter {k}, dirs: {dirs}")

        for r in range(1,4):
            comm.Isend([dummy, MPI.DOUBLE], dest=r, tag=TAG_SEND_U)

        for r in range(1,4):
            u[r-1] = comm.recv(source=r, tag=TAG_U_FROM_ROOM)
            print(f"from {r}: shape of u: {u[r-1].shape}")

        plotting.plot_u(u, show_labels=False)


    # Send shutdown commands
    for r in range(1,4):
        comm.Send([dummy, MPI.DOUBLE], dest=r, tag=TAG_STOP)


elif rank == 1:
    r1 = room.Room1(comm, gridsize, T)
    r1.run(w)

elif rank == 2:
    r2 = room.Room2(comm, gridsize, T)
    r2.run(w)

elif rank == 3:
    r3 = room.Room3(comm, gridsize, T)
    r3.run(w)

print(f"Process {rank} finished")
