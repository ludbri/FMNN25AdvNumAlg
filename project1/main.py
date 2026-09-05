import numpy as np
from scipy.linalg import solve
from itertools import product

# parameters
roomsizes = np.array([[1,1],[2,1],[1,1]])
gridsize = 3
h = 1 / gridsize
# fixed wall temperatures:
T_w = 15
T_h = 40
T_wf = 5

# starting with room 1
room_idx = 2
N = tuple(gridsize*r + 1 for r in roomsizes[room_idx])
n_vars = N[0] * N[1]
u = np.empty(n_vars, dtype='d')
A = np.zeros((n_vars, n_vars), dtype='d')
rhs = np.zeros_like(u)


# constraints for the interior points (rhs=0):
# boundary points will later be overwritten based on the room
# center
k = 0
A += np.diagflat(-4*np.ones(n_vars-abs(k)),k)
# i-1
k = -1
A += np.diagflat(np.ones(n_vars-abs(k)),k)
# i+1
k = 1
A += np.diagflat(np.ones(n_vars-abs(k)),k)
# j-1
k = -N[1]
A += np.diagflat(np.ones(n_vars-abs(k)),k)
# j+1
k = N[1]
A += np.diagflat(np.ones(n_vars-abs(k)),k)

# When A is reshaped as (*N,*N), the indexing is [rhs_i, rhs_j, u_i, u_j]
# i.e., [0,0,0,:] 
A = A.reshape(*N,*N)
rhs = rhs.reshape(*N)



# room dependent boundary conditions
if room_idx == 0:
    # incoming Neumann boundary conditions
    neu_1 = np.array([10]*(N[0]-2))

    # East wall (first in order to later override corners as fixed values)
    A = A.reshape(n_vars,n_vars)
    for i, x in enumerate(range(1,N[0]-1)):
        idx_x = x * N[1] + (N[1] - 1)
        A[idx_x, :] = 0
        A[idx_x, idx_x] = 1
        A[idx_x, idx_x+1] = -1  # one step right
        # TODO: should this flip sign (to +)? the message is outgoing derivative, receiver uses incoming?
        rhs[x,-1] = h * neu_1[i]  # one step right (neumann condition)
    A = A.reshape(*N,*N)

    # North wall
    A[0, :, :, :] = 0
    A[0, :, 0, :] = np.eye(N[1])
    rhs[0, :] = T_w

    # South wall
    A[-1, :, :, :] = 0
    A[-1, :, -1, :] = np.eye(N[1])
    rhs[-1, :] = T_w

    # West wall
    A[:,0,:,:] = 0
    A[:,0,:,0] = np.eye(N[0])
    rhs[:,0] = T_h




elif room_idx == 1:
    # half vertical wall size:
    n_half = N[0] // 2 + 1
    # incoming Dirichlet boundary conditions
    dir_1 = np.array([0]*(n_half-1))
    dir_2 = np.array([0]*(n_half-1))

    # Note: east/west before north/south to override corners with fixed heat walls

    # West wall top
    A[:n_half,0, :,:] = 0
    A[:n_half,0, :n_half,0] = np.eye(n_half)
    rhs[:n_half,0] = T_w
    # West wall bottom
    A[n_half:,0, :,:] = 0
    A[n_half:,0, n_half:,0] = np.eye(n_half-1)
    rhs[n_half:,0] = dir_1  # this is where Dirichlet condition from room 1 is used

    # East wall top
    A[:n_half-1,-1,:,:] = 0
    A[:n_half-1,-1,:n_half-1,-1] = np.eye(n_half-1)
    rhs[:n_half-1,-1] = dir_2  # this is where Dirichlet condition from room 3 is used
    # East wall bottom
    A[n_half-1:,-1,:,:] = 0
    A[n_half-1:,-1,n_half-1:,-1] = np.eye(n_half)
    rhs[n_half-1:,-1] = T_w

    # North wall
    A[0, :, :, :] = 0
    A[0, :, 0, :] = np.eye(N[1])
    rhs[0, :] = T_h

    # South wall
    A[-1, :, :, :] = 0
    A[-1, :, -1, :] = np.eye(N[1])
    rhs[-1, :] = T_wf




if room_idx == 2:
    # incoming Neumann boundary conditions
    neu_2 = np.array([100]*(N[0]-2))

    A = A.reshape(n_vars,n_vars)
    for i, x in enumerate(range(1,N[0]-1)):
        idx_x = x * N[1]
        A[idx_x, :] = 0
        A[idx_x, idx_x] = 1
        A[idx_x, idx_x+1] = -1  # one step left
        # TODO: should this flip sign (to +)? the message is outgoing derivative, receiver uses incoming?
        rhs[x,0] = h * neu_2[i]  # Neumann condition
    A = A.reshape(*N,*N)

    # West wall (first in order to later override corners as fixed values)
    # A[:,0,:,:] = 0
    # A[:,0,:,0] = np.eye(N[0])
    # rhs[:,0] = 0

    # North wall
    A[0, :, :, :] = 0
    A[0, :, 0, :] = np.eye(N[1])
    rhs[0, :] = T_w

    # South wall
    A[-1, :, :, :] = 0
    A[-1, :, -1, :] = np.eye(N[1])
    rhs[-1, :] = T_w

    # East wall
    A[:,-1,:,:] = 0
    A[:,-1,:,-1] = np.eye(N[0])
    rhs[:,-1] = T_h




# reshape back before solving
A = A.reshape(n_vars,n_vars)
rhs = rhs.reshape(n_vars)
# Note: corner points do not matter, as they are only attached to neighbours with fixed heat.

# printouts for dev
print(np.arange(n_vars,dtype='d').reshape(N[0],-1))
print(np.arange(n_vars,dtype='d'))
print(A.reshape(n_vars,n_vars))
print(np.arange(n_vars,dtype='d'))
print(rhs.reshape(1,-1))

sol = solve(A, rhs)
print(sol.reshape(*N))


# depending on room, compute the output boundary condition
if room_idx == 0:
    ...
elif room_idx == 1:
    ...
elif room_idx == 2:
    ...

