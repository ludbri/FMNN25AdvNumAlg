import numpy as np
import room
from mpi4py import MPI

# TODO: move to MPI
# TODO: move to sparse matrices?

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
# processes 1,2,3 have their own room
# process 0 coordinates? or at least summarizes and plots.

rank = comm.Get_rank()

# using tags to communicate whether or not to stop.
# Idea suggested by copilot.
TAG_STOP = 0

if rank == 0:
    # coordination
    dirs = np.zeros((2,c_size), dtype='d')
    neus = np.zeros((2,c_size), dtype='d')

    for k in range(5):
        # Solve room 2:
        comm.Send([dirs, MPI.DOUBLE], dest=2, tag=1)
        comm.Recv([neus, MPI.DOUBLE], source=2, tag=2)

        # Solve room 1 and 3:
        comm.Isend([neus[0], MPI.DOUBLE], dest=1, tag=1)
        comm.Isend([neus[1], MPI.DOUBLE], dest=3, tag=2)

        comm.Irecv([dirs[0], MPI.DOUBLE], source=1, tag=3)
        comm.Recv([dirs[1], MPI.DOUBLE], source=3, tag=4)

        print(f"iter {k}, dirs: {dirs}")

    # Send shutdown commands
    dummy = np.empty(1, dtype='d')
    for r in range(1,4):
        comm.Send([dummy, MPI.DOUBLE], dest=r, tag=TAG_STOP)


elif rank == 1:
    # room 1
    r1 = room.Room1(comm, gridsize, T)

    dir = np.zeros(c_size, dtype='d')
    neu = np.zeros(c_size, dtype='d')
    status = MPI.Status()
    while True:
        comm.Probe(source=0, tag=MPI.ANY_TAG, status=status)

        if status.Get_tag() == TAG_STOP:
            dummy = np.empty(1, dtype='d')
            comm.Recv(dummy, source=0, tag=TAG_STOP)
            break
    
        comm.Recv([neu, MPI.DOUBLE], source=0, tag=1)

        dir = r1.solve(neu)

        comm.Send([dir, MPI.DOUBLE], dest=0, tag=3)


elif rank == 2:
    # room 2
    dirs = np.zeros((2,c_size), dtype='d')
    neus = np.zeros((2,c_size), dtype='d')

    r2 = room.Room2(comm, gridsize, T)
    status = MPI.Status()
    while True:
        comm.Probe(source=0, tag=MPI.ANY_TAG, status=status)

        if status.Get_tag() == TAG_STOP:
            dummy = np.empty(1, dtype='d')
            comm.Recv(dummy, source=0, tag=TAG_STOP)
            break

        comm.Recv([dirs, MPI.DOUBLE], source=0, tag=1)

        neus = r2.solve(dirs)

        comm.Send([neus, MPI.DOUBLE], dest=0, tag=2)


elif rank == 3:
    # room 3
    dir = np.zeros(c_size, dtype='d')
    neu = np.zeros(c_size, dtype='d')

    r3 = room.Room3(comm, gridsize, T)
    status = MPI.Status()
    while True:
        comm.Probe(source=0, tag=MPI.ANY_TAG, status=status)

        if status.Get_tag() == TAG_STOP:
            dummy = np.empty(1, dtype='d')
            comm.Recv(dummy, source=0, tag=TAG_STOP)
            break
    
        comm.Recv([neu, MPI.DOUBLE], source=0, tag=2)

        dir = r3.solve(neu)

        comm.Send([dir, MPI.DOUBLE], dest=0, tag=4)


print(f"Process {rank} finished")
