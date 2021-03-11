import multiprocessing as mp
from itertools import product

def my_func(x, y, z):
  print(mp.current_process())
  return [x**x, y, z/2]

def sss():
  pool = mp.Pool(mp.cpu_count())
  result = pool.starmap(my_func, product([1,2,3], [2,3,4], [3,4,5]))
  print(result)
  

if __name__ == "__main__":
  sss()
  