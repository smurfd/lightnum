import random as rnd
from lightnum.array import array, ndarray, asarray
from lightnum.helper import *

class random():
  def seed(x, dtype=int32): return rnd.seed(x)
  def randint(x, dtype=int32): return rnd.randint(x)
  def randn(*args, dtype=int32, ret=[]):
    if isinstance(args, tuple) and len(args) == 1:
      if len(args[0]) != 1: [ret.append(i) for j in range(len(args[0])) for i in range(args[0][j]) if i != j]
      else: [ret.append(i) for i in range(len(args[0]))]
    else: [ret.append(i) for i in args]
    return ndarray(helper.looper(helper, ret, fill=rnd.random(), noloop=True))
  def random(size, dtype=float32): return random.randn(size)
  def rand(*args, dtype=int32): return random.randn(args, dtype=dtype)
  def choise(x, dtype=int32): return rnd.choise(x)
  def RandomState(x, dtype=int32): return rnd.RandomState(x)
  def default_rng(x, dtype=int32): return rnd.default_rng(x)
