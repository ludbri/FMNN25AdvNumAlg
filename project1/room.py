import numpy as np
from mpi4py import MPI
from scipy.linalg import solve

from tags import *




class Room:
    """Base class for a room"""

    def __init__(self, comm: MPI.Comm, roomsize: tuple[int,int], gridsize: int, msg_shape: tuple[int,int]):
        """Initialize parameters and base matrices"""
        self.comm = comm
        self.roomsize = roomsize
        self.h = 1/gridsize
        self.N = tuple(gridsize*r + 1 for r in roomsize)
        self.n_vars = self.N[0] * self.N[1]
        self.msg_shape = msg_shape  # shape of boundary condition messages

        # The matrices are stored as the standard 2-D and 1-D shapes used in solving.
        # i.e. A.shape(n,n), u.shape=b.shape(n,1)
        self.A = np.zeros((self.n_vars, self.n_vars), dtype='d')
        self.u = np.zeros((self.n_vars,1), dtype='d')
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

    def set_boundary(self, _cond: np.array):
        """Set the incoming boundary condition"""
        raise NotImplementedError("implemented in subclasses")

    def get_boundary(self):
        """Return outgoing boundary condition(s)"""
        raise NotImplementedError("implemented in subclasses")

    def solve(self, cond: np.array | None) -> np.array:
        """Solve the equation system and return the output condition"""
        if cond is not None:
            self.set_boundary(cond)
        self.u = solve(self.A, self.b)
        return self.get_boundary()

    def run(self, w):
        """
        Performs the execution loop.

        The loop is exited when a signal with tag=TAG_STOP is received
        """
        status = MPI.Status()

        inb_cond = np.empty(self.msg_shape, dtype='d')
        outb_cond = np.empty_like(inb_cond)

        while True:
            self.comm.Probe(source=0, tag=MPI.ANY_TAG, status=status)

            if status.Get_tag() == TAG_STOP:
                dummy = np.empty(1, dtype='d')
                self.comm.Recv(dummy, source=0, tag=TAG_STOP)
                break

            elif status.Get_tag() == TAG_SEND_U:
                dummy = np.empty(1, dtype='d')
                self.comm.Recv(dummy, source=0, tag=TAG_SEND_U)
                self.comm.send(self.u, dest=0, tag=TAG_U_FROM_ROOM)
                continue
        
            self.comm.Recv([inb_cond, MPI.DOUBLE], source=0, tag=TAG_TO_ROOM)

            old_u = self.u.copy()
            outb_cond = self.solve(inb_cond)

            if self.comm.Get_rank() == 2:
                print(f"u2:\n{self.u.reshape(self.N)}")

            # relax
            self.u = w*self.u + (1-w)*old_u
            # print(f"{self.comm.Get_rank()} u shape: {self.u.shape}")

            self.comm.Send([outb_cond, MPI.DOUBLE], dest=0, tag=TAG_FROM_ROOM)



class Room1(Room):
    def __init__(self, comm: MPI.Comm, gridsize, T):
        super().__init__(comm, [1,1], gridsize, msg_shape=(gridsize-1,))
        self.u[:] = T['w']

        self.b = self.b.reshape(*self.N)

        # East wall (first in order to later override corners as fixed values)
        # the rhs is the Neumann condition and initially 0
        self.A = self.A.reshape(*self.N,*self.N)
        for x in range(1,self.N[0]-1):
            # self.A[x,-1, :,:] = 0
            # self.A[x,-1, x,-1] = -1
            # self.A[x,-1, x,-2] = 1  # one step left
            # flip:
            self.A[x,-1, :,:] = 0
            self.A[x,-1, x,-1] = -3
            self.A[x,-1, x-1,-1] = 1
            self.A[x,-1, x+1,-1] = 1
            self.A[x,-1, x,-2] = 1

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

        # print(self.A)
        # raise Exception

    def set_boundary(self, neumann_cond):
        # self.b.reshape(*self.N)[1:-1, -1] = self.h * neumann_cond
        self.b.reshape(*self.N)[1:-1, -1] = - self.h * neumann_cond

    def get_boundary(self):
        dir_1 = self.u.reshape(*self.N)[1:-1,-1].copy()
        return dir_1



class Room2(Room):
    def __init__(self, comm: MPI.Comm, gridsize, T):
        super().__init__(comm, [2,1], gridsize, msg_shape=(2, gridsize-1))
        self.u[:] = T['w']

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
        # Note: initial Dirichlet conditions are 0

        # East wall top
        self.A[1:self.n_half-1,-1, :,:] = 0
        self.A[1:self.n_half-1,-1, 1:self.n_half-1,-1] = np.eye(self.c_size)
        # Note: initial Dirichlet conditions are 0
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
        A_nc1 = np.zeros((self.c_size,*self.N),dtype='d')
        A_nc2 = np.zeros_like(A_nc1)
        for i, x in enumerate(range(1,self.c_size+1)):
            x1 = i + self.n_half  # offset to the lower half of the left wall
            # center
            # A_nc1[i, x1,0] = A_nc2[i, x,-1] = -3
            # # step left
            # A_nc2[i, x,-2] = 1
            # # step right
            # A_nc1[i, x1,1] = 1
            # # step up
            # A_nc1[i, x1-1,0] = 1
            # A_nc2[i, x-1,-1] = 1
            # # step down
            # A_nc1[i, x1+1,0] = 1
            # A_nc2[i, x+1,-1] = 1
            # flip!
            A_nc1[i, x1,0] = A_nc2[i, x,-1] = -1
            A_nc1[i, x1,1] = A_nc2[i, x,-2] = 1
        
        A_nc1 = A_nc1.reshape(-1,self.n_vars)
        A_nc2 = A_nc2.reshape(-1,self.n_vars)
        self.A_nc = np.array([A_nc1, A_nc2])

    def set_boundary(self, dirichlet_conds):
        self.b.reshape(*self.N)[self.n_half:-1,0] = dirichlet_conds[0]
        self.b.reshape(*self.N)[1:self.n_half-1,-1] = dirichlet_conds[1]

    def get_boundary(self):
        neus = self.A_nc @ self.u
        return neus


class Room3(Room):
    def __init__(self, comm: MPI.Comm, gridsize, T):
        super().__init__(comm, [1,1], gridsize, msg_shape=(gridsize-1,))
        self.u[:] = T['w']

        self.b = self.b.reshape(*self.N)

        # West wall (first in order to later override corners as fixed values)
        # the rhs is the Neumann condition and initially 0
        self.A = self.A.reshape(*self.N,*self.N)
        for x in range(1,self.N[0]-1):
            # self.A[x,0, :,:] = 0
            # self.A[x,0, x,0] = -1
            # self.A[x,0, x,1] = 1  # one step right
            # flip
            self.A[x,0, :,:] = 0
            self.A[x,0, x,0] = -3
            self.A[x,0, x,1] = 1  # one step right
            self.A[x,0, x-1,1] = 1  # one step up
            self.A[x,0, x+1,1] = 1  # one step left

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
        # self.b.reshape(*self.N)[1:-1, 0] = self.h * neumann_cond
        self.b.reshape(*self.N)[1:-1, 0] = - self.h * neumann_cond

    def get_boundary(self):
        dir_2 = self.u.reshape(*self.N)[1:-1,0].copy()
        return dir_2

