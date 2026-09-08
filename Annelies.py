import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from mpi4py import MPI

comm = MPI.Comm.Clone(MPI.COMM_WORLD)
rank = comm.Get_rank()

# problem parameters
gridsize = 3
N_i = gridsize+1
N = (N_i,N_i)
dx = 1/gridsize
omega = 0.8
iterations = 1

T_w = 15.0
T_h = 40.0
T_wh = 5

# init grids w default
if rank == 0:
    u = T_w * np.ones(N)
    u[0,:] = T_h
elif rank == 1:
    u = T_w * np.ones((N_i, N_i*2-1))
    u[:,0] = T_wh
    u[:,-1] = T_h
elif rank == 2:
    u = T_w * np.ones(N)
    u[N_i-1, :] = T_h


def build_matrices():
    # only interior nodes
    A2 = np.zeros(((gridsize-1)*(gridsize*2-1),
                   (gridsize-1)*(gridsize*2-1)))
    for j in range(gridsize*2-1):
        for i in range(gridsize-1):
            row = i + j * (gridsize-1)
            A2[row, row] = 4.0
            if i>0:
                A2[row, row-1]=-1.0
            if i<gridsize-2:
                A2[row, row+1]=-1.0
            if j>0:
                A2[row, row-(gridsize-1)]=-1.0
            if j<gridsize*2-2:
                A2[row, row+(gridsize-1)]=-1.0

    # interior and right boundary
    A1 = np.zeros(((gridsize)*(gridsize-1),
                   (gridsize)*(gridsize-1)))
    for j in range(gridsize-1):
        for i in range(gridsize):
            row = i + j * gridsize
            A1[row, row] = 4.0
            if i>0:
                if i<gridsize-1:
                    A1[row, row-1]=-1.0
                else:
                    A1[row, row-1]=-2.0  # ghost modifier
            if i<gridsize-1:
                A1[row, row+1]=-1.0
            if j>0:
                A1[row, row - gridsize]=-1.0
            if j<gridsize-2:
                A1[row, row + gridsize]=-1.0

    # interior and left boundary
    A3 = np.zeros(((gridsize)*(gridsize-1),
                   (gridsize)*(gridsize-1)))
    for j in range(gridsize-1):
        for i in range(gridsize):
            row = i + j * gridsize
            A3[row, row] = 4.0
            if i>0:
                A3[row, row-1]=-1.0
            if i<gridsize-1:
                if i>0:
                    A3[row, row+1]=-1.0
                else:
                    A3[row, row+1]=-2.0  # ghost modifier
            if j>0:
                A3[row, row - gridsize]=-1.0
            if j<gridsize-2:
                A3[row, row + gridsize]=-1.0

    return A1, A2, A3

A1, A2, A3 = build_matrices()

# iter loop
for k in range(iterations):
    if rank == 1:
        # receive bounds from 1 and 3
        G1 = comm.recv(source=0,tag=1)
        G2 = comm.recv(source=2,tag=2)

        print("\n",k)
        print(f"inc G1, dir1:\n{G1}")
        print(f"inc G2, dir2:\n{G2}")

        # build RHS for domain 2
        b2 = np.zeros((gridsize-1)*(gridsize*2-1))
        for j in range(gridsize*2-1):
            for i in range((gridsize-1)):
                row = i + j * (gridsize-1)
                if j == 0:
                    b2[row] += u[i+1, 0]
                if j == gridsize*2-2:
                    b2[row] += u[i+1, gridsize*2]
                if i == 0:
                    if j < gridsize-1:
                        b2[row] += G1[j+1]
                    else:
                        T_w
                if i == gridsize-2:
                    if j < gridsize-1:
                        b2[row] += T_w
                    else:
                        b2[row] += G2[j-(gridsize-1)]

        print(f"A2:\n{A2}")
        print(f"b2:\n{b2}")

        u2_star_vec = solve(A2, b2)
        print(f"u2:\n{u2_star_vec}")
        print(f"A2@u2:\n{A2@u2_star_vec}")

        u2_star = u2_star_vec.reshape(gridsize*2-1,gridsize-1).T

        print(f"iter {k}, U2 prerelax my seq:\n{u2_star.T[::-1]}")

        # send gradients
        g1 = u2_star[0,:gridsize-1] - G1[1:gridsize]
        g2 = G2[1:gridsize] - u2_star[gridsize-2, gridsize-1:gridsize*2-2]
        comm.send(g1, dest=0, tag=11)
        comm.send(g2, dest=2, tag=22)

        # relax
        u[1:gridsize, 1:gridsize*2] = omega * u2_star + (1-omega) * u[1:gridsize, 1:gridsize*2]


    elif rank == 0:
        comm.send(u[gridsize, :], dest=1, tag=1)
        g1 = comm.recv(source=1, tag=11)

        # print("\n",k)
        # print(f"inc g1, neu1:\n{g1}")

        b1 = np.zeros(gridsize*(gridsize-1))
        for j in range(gridsize-1):
            for i in range(gridsize):
                row = i + j * gridsize
                if j == 0: 
                    b1[row] += u[i+1, 0]
                if j == gridsize-2: 
                    b1[row] += u[i+1, gridsize]
                if i == 0: 
                    b1[row] += u[0, j+1]
                if i == gridsize-1: 
                    b1[row] += 2.0 * g1[j]

        u1_star = solve(A1, b1).reshape((gridsize-1, gridsize)).T
        u[1:gridsize+1, 1:gridsize] = omega * u1_star + (1 - omega) * u[1:gridsize+1, 1:gridsize]


    elif rank == 2:
        comm.send(u[0, :], dest=1, tag=2)
        g2 = comm.recv(source=1, tag=22)

        # print("\n",k)
        # print(f"inc g2, neu2:\n{g2}")
        
        b3 = np.zeros(gridsize*(gridsize-1))
        for j in range((gridsize-1)):
            for i in range(gridsize):
                row = i + j * gridsize
                if j == 0:
                    b3[row] += u[i, 0]
                if j == gridsize-2:
                    b3[row] += u[i, gridsize]
                if i == (gridsize-1):
                    b3[row] += u[gridsize, j+1]
                if i == 0:
                    b3[row] -= 2.0 * g2[j]

        u3_star = solve(A3, b3).reshape((gridsize-1, gridsize)).T
        u[0:gridsize, 1:gridsize] = omega * u3_star + (1 - omega) * u[0:gridsize, 1:gridsize]

# Gather and Plot
if rank == 0:
    u2 = comm.recv(source=1, tag=99)
    u3 = comm.recv(source=2, tag=99)
    
    # Construct full apartment grid (61 x 41)
    full_grid = np.full((gridsize*2+1, gridsize*3+1), np.nan)
    full_grid[0:gridsize*2+1, gridsize:gridsize*2+1] = u2.T
    full_grid[0:gridsize+1, 0:gridsize+1] = u.T
    full_grid[gridsize:gridsize*2+1, gridsize*2:gridsize*3+1] = u3.T
    
    plt.figure(figsize=(10, 6))
    plt.imshow(full_grid, origin='lower', cmap='hot', vmin=5, vmax=50)
    plt.colorbar(label='Temperature (°C)')
    plt.title('2-Room Apartment Temperature Distribution (MPI)')
    plt.xlabel('x (nodes)')
    plt.ylabel('y (nodes)')
    plt.show()
    
    # print(A1, "\n")
    # print(A2, "\n")
    # print(A3)
else:
    comm.send(u, dest=0, tag=99)

