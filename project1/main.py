import numpy as np
import room

# TODO: move to MPI
# TODO: move to sparse matrices?

# parameters
roomsizes = np.array([[1,1],[2,1],[1,1]])
gridsize = 3
h = 1 / gridsize
# fixed wall temperatures:
T = {"w": 15,
     "h": 40,
     "wf": 5}


r1 = room.Room1(0, roomsizes[0], gridsize, T)
print(r1.solve(10))

r2 = room.Room2(1, roomsizes[1], gridsize, T)
print(r2.solve([0,0]))

r3 = room.Room3(2, roomsizes[2], gridsize, T)
print(r3.solve(100))