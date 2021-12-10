from mpi4py import MPI
import numpy as np
import scipy.stats as sts
import time
import health_sim

def sim_lifetimes_parallel(S, T, rho, mu, sigma):
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()  
  size = comm.Get_size()

#   np.random.seed(rank)
  strt_time = time.time()

  S_proc = int(S / size)
  np.random.seed(rank)
  eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, S_proc)) # 4160 weeks per life, 1000 lives
  z_mat = health_sim.sim_lifetimes(S_proc, T, rho, mu, eps_mat) 

  z_mat_all = None
  if rank == 0:
    z_mat_all = np.zeros([T, S_proc * size], dtype='float64')
  comm.Gather(z_mat, z_mat_all, root = 0)
  parallel_time = time.time() - strt_time
  if rank == 0:
   print(str(size) + ',' + str(parallel_time) + '\n')

def main():
  rho = 0.5
  mu = 3.0
  sigma = 1.0
  S = 1000
  T = int(4160)
  sim_lifetimes_parallel(S, T, rho, mu, sigma)


if __name__ == '__main__':
  main()