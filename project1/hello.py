import mpi4py.MPI as MPI

comm = MPI.Comm.Clone(MPI.COMM_WORLD)

r = comm.Get_rank()
n = comm.Get_size()

print(f"Hello from rank {r} out of {n}")