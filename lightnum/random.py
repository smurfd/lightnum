from lightnum.array import array, ndarray, asarray
from lightnum.helper import helper
from lightnum.dtypes import int32, float32
import random as rnd

class random():
  class default_rng():
    def __init__(self, x): self.x = x
    def __call__(self): return rnd.default_rng(self.x)
  def seed(x, dtype=int32): return rnd.seed(x)
  def randint(x, y=None, size=None, dtype=int32):
    if size and y: return [rnd.randint(x, y) for i in range(size)]
    if size: return [rnd.randint(x) for i in range(size)]
    if y: return rnd.randint(x, y)
    return rnd.randint(x)
  def randn(*args, dtype=int32, ret=[]):
    if isinstance(args, tuple) and len(args) == 1:
      if len(args[0]) != 1: [ret.append(i) for j in range(len(args[0])) for i in range(args[0][j]) if i != j]
      else: [ret.append(i) for i in range(len(args[0]))]
    else: [ret.append(i) for i in args]
    return ndarray(helper.looper_empty(ret, fill=rnd.random(), dtype=float32))
  def random(size, dtype=float32): return random.randn(size)
  def rand(*args, dtype=int32): return random.randn(args, dtype=dtype)
  def choise(x, dtype=int32): return rnd.choise(x)
  def RandomState(x, dtype=int32): return rnd.RandomState(x)
