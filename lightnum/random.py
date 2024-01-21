from lightnum.array import array, ndarray, asarray
from lightnum.helper import helper
from lightnum.dtypes import int32, float32
import random as rnd

class random():
  class default_rng():
    def __init__(self, x): self.x = x
    def __call__(self): return rnd.default_rng(self.x)
    def random(self,size, dtype=float32): return random.randn(size)
  def seed(x, dtype=int32): return rnd.seed(x)
  class randint():
    def __init__(self, x, y=None, size=None, dtype=int32):
      if size and y: self.ret = [rnd.randint(x, y) for i in range(size)]
      if size: self.ret = [rnd.randint(0, x) for i in range(size)]
      if y: self.ret = rnd.randint(x, y)
    def __call__(self): return rnd.randint(self.x)
    def tolist(self): return self.ret
  def randn(*args, dtype=int32, ret=[]):
    if isinstance(args, tuple) and len(args) == 1:
      if len(args[0]) != 1: [ret.append(i) for j in range(len(args[0])) for i in range(args[0][j]) if i != j]
      else: [ret.append(i) for i in range(len(args[0]))]
    else: [ret.append(i) for i in args]
    return ndarray(helper.looper_empty(ret, fill=rnd.random(), dtype=float32))
  def random(size, dtype=float32): return random.randn(size)
  def rand(*args, dtype=int32): return random.randn(args, dtype=dtype)
  def RandomState(x, dtype=int32): return rnd.RandomState(x)
  class uniform():
    def __init__(self, x, y, size): self.ret = [rnd.randint(x, y) for _ in range(size)] if size else rnd.randint(x, y)
    def tolist(self): return self.ret
  class choice():
    def __init__(self, x, size=0, dtype=int32): self.ret = [rnd.choice(x) for _ in range(size)] if size else rnd.choice(x)
    def tolist(self): return self.ret
