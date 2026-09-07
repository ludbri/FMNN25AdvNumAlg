import numpy as np
from scipy.linalg import solve


class Room:
    """Base class for a room"""

    def __init__(self, idx: int, roomsize: tuple[int,int], gridsize: int):
        """Initialize parameters and base matrices"""
        self.idx = idx
        self.roomsize = roomsize
        self.h = 1/gridsize
        self.N = tuple(gridsize*r + 1 for r in roomsize)
        self.n_vars = self.N[0] * self.N[1]

        # The matrices are stored as the standard 2-D and 1-D shapes used in solving.
        # i.e. A.shape(n,n), u.shape=b.shape(n,1)
        self.A = np.zeros((self.n_vars, self.n_vars), dtype='d')
        self.u = np.empty(self.n_vars, dtype='d')
        self.b = np.zeros_like(self.u)
        
        # constraints for the interior points (rhs=0):
        # boundaries will later be overwritten based on the specific room
        # center
        k = 0
        self.A += np.diagflat(-4*np.ones(self.n_vars-abs(k)),k)
        # i-1
        k = -1
        self.A += np.diagflat(np.ones(self.n_vars-abs(k)),k)
        # i+1
        k = 1
        self.A += np.diagflat(np.ones(self.n_vars-abs(k)),k)
        # j-1
        k = -self.N[1]
        self.A += np.diagflat(np.ones(self.n_vars-abs(k)),k)
        # j+1
        k = self.N[1]
        self.A += np.diagflat(np.ones(self.n_vars-abs(k)),k)

    def set_boundary(self, cond: np.array):
        """Set the incoming boundary condition"""
        raise NotImplementedError("implemented in subclasses")

    def get_boundary(self):
        """Return outgoing boundary condition(s)"""
        raise NotImplementedError("implemented in subclasses")

    def solve(self, cond: np.array) -> np.array:
        """Solve the equation system and return the output condition"""
        self.set_boundary(cond)
        self.u = solve(self.A.reshape(self.n_vars, self.n_vars),
                       self.b.reshape(self.n_vars, 1))
        return self.get_boundary()



class Room1(Room):
    def __init__(self, idx, roomsize, gridsize, T):
        super().__init__(idx, roomsize, gridsize)

        self.b = self.b.reshape(*self.N)

        # East wall (first in order to later override corners as fixed values)
        # self.A = self.A.reshape(self.n_vars,self.n_vars)
        # for x in range(1,self.N[0]-1):
        #     idx_x = (x+1) * self.N[1] - 1
        #     self.A[idx_x, :] = 0
        #     self.A[idx_x, idx_x] = -1
        #     self.A[idx_x, idx_x-1] = 1  # one step left
        # # the assumed starting condition for neumann is 0:
        # # self.b[1:-1,-1] = self.h * 0  # one step right (neumann condition)
        self.A = self.A.reshape(*self.N,*self.N)
        for x in range(1,self.N[0]-1):
            self.A[x,-1, :,:] = 0
            self.A[x,-1, x,-1] = -1
            self.A[x,-1, x,-2] = 1  # one step left

        # North wall
        self.A[0, :, :, :] = 0
        self.A[0, :, 0, :] = np.eye(self.N[1])
        self.b[0, :] = T["w"]

        # South wall
        self.A[-1, :, :, :] = 0
        self.A[-1, :, -1, :] = np.eye(self.N[1])
        self.b[-1, :] = T["w"]

        # West wall
        self.A[:,0,:,:] = 0
        self.A[:,0,:,0] = np.eye(self.N[0])
        self.b[:,0] = T["h"]

        self.A = self.A.reshape(self.n_vars,self.n_vars)
        self.b = self.b.reshape(self.n_vars,1)

    def set_boundary(self, neumann_cond):
        self.b.reshape(*self.N)[1:-1, -1] = self.h * neumann_cond

    def get_boundary(self):
        dir_1 = self.u.reshape(*self.N)[1:-1,-1]
        return dir_1



class Room2(Room):
    def __init__(self, idx, roomsize, gridsize, T):
        super().__init__(idx, roomsize, gridsize)

        # half vertical wall size:
        self.n_half = self.N[0] // 2 + 1
        # condition size
        self.c_size = self.n_half - 2

        self.b = self.b.reshape(*self.N)
        self.A = self.A.reshape(*self.N,*self.N)

        # Note: east/west before north/south to override corners with fixed heat walls
        # West wall top
        self.A[:self.n_half,0, :,:] = 0
        self.A[:self.n_half,0, :self.n_half,0] = np.eye(self.n_half)
        self.b[:self.n_half,0] = T["w"]
        # West wall bottom
        self.A[self.n_half:-1,0, :,:] = 0
        self.A[self.n_half:-1,0, self.n_half:-1, 0] = np.eye(self.c_size)
        # Note: initial conditions are 0
        # self.b[self.n_half:-1,0] = 0  # this is where Dirichlet condition from room 1 is used

        # East wall top
        self.A[1:self.n_half-1,-1, :,:] = 0
        self.A[1:self.n_half-1,-1, 1:self.n_half-1,-1] = np.eye(self.c_size)
        # Note: initial conditions are 0
        # self.b[1:self.n_half-1,-1] = 0  # this is where Dirichlet condition from room 3 is used
        # East wall bottom
        self.A[self.n_half-1:,-1, :,:] = 0
        self.A[self.n_half-1:,-1, self.n_half-1:,-1] = np.eye(self.n_half)
        self.b[self.n_half-1:,-1] = T["w"]

        # North wall
        self.A[0, :, :, :] = 0
        self.A[0, :, 0, :] = np.eye(self.N[1])
        self.b[0, :] = T["h"]

        # South wall
        self.A[-1, :, :, :] = 0
        self.A[-1, :, -1, :] = np.eye(self.N[1])
        self.b[-1, :] = T["wf"]

        self.A = self.A.reshape(self.n_vars,self.n_vars)
        self.b = self.b.reshape(self.n_vars,1)

        # Matrices for outgoing conditions (Neumann - derivative)
        self.A_nc1 = np.zeros((self.c_size,*self.N),dtype='d')
        self.A_nc2 = self.A_nc1.copy()
        for i, x in enumerate(range(1,self.c_size+1)):
            x1 = i + self.n_half  # offset to the lower half of the left wall
            # center
            self.A_nc1[i, x1,0] = self.A_nc2[i, x,-1] = -3
            # step left
            self.A_nc2[i, x,-2] = 1
            # step right
            self.A_nc1[i, x1,1] = 1
            # step up
            self.A_nc1[i, x1-1,0] = 1
            self.A_nc2[i, x-1,-1] = 1
            # step down
            self.A_nc1[i, x1+1,0] = 1
            self.A_nc2[i, x+1,-1] = 1
        self.A_nc1 = self.A_nc1.reshape(-1,self.n_vars)
        self.A_nc2 = self.A_nc2.reshape(-1,self.n_vars)

    def set_boundary(self, dirichlet_conds):
        self.b.reshape(*self.N)[self.n_half:-1,0] = dirichlet_conds[0]
        self.b.reshape(*self.N)[1:self.n_half-1,-1] = dirichlet_conds[1]

    def get_boundary(self):
        neu_1 = self.A_nc1 @ self.u
        neu_2 = self.A_nc2 @ self.u
        return neu_1, neu_2


class Room3(Room):
    def __init__(self, idx, roomsize, gridsize, T):
        super().__init__(idx, roomsize, gridsize)

        self.b = self.b.reshape(*self.N)

        # West wall (first in order to later override corners as fixed values)
        # initial neumann condition is zero
        # # self.b[1:-1,0] = self.h * 0  # one step left (neumann condition)
        self.A = self.A.reshape(*self.N,*self.N)
        for x in range(1,self.N[0]-1):
            self.A[x,0, :,:] = 0
            self.A[x,0, x,0] = -1
            self.A[x,0, x,1] = 1  # one step right

        # North wall
        self.A[0, :, :, :] = 0
        self.A[0, :, 0, :] = np.eye(self.N[1])
        self.b[0, :] = T["w"]

        # South wall
        self.A[-1, :, :, :] = 0
        self.A[-1, :, -1, :] = np.eye(self.N[1])
        self.b[-1, :] = T["w"]

        # East wall
        self.A[:,-1,:,:] = 0
        self.A[:,-1,:,-1] = np.eye(self.N[0])
        self.b[:,-1] = T["h"]

        self.A = self.A.reshape(self.n_vars,self.n_vars)
        self.b = self.b.reshape(self.n_vars,1)

    def set_boundary(self, neumann_cond):
        self.b.reshape(*self.N)[1:-1, 0] = self.h * neumann_cond

    def get_boundary(self):
        dir_2 = self.u.reshape(*self.N)[1:-1,0]
        return dir_2

