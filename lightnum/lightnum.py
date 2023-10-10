from lightnum.array import array, ndarray, asarray
from lightnum.random import random
from lightnum.testing import testing
from lightnum.helper import helper
from lightnum.dtypes import int32,float32
import random as rnd
import array as arr
import copy as cp
import builtins
import math
# TODO: make dtype=xxx do something
# TODO: class lib
# TODO: add functionallity to functions with pass

# math
def log(x, dtype=int32): return helper.looper(helper, x, call=math.log, dtype=dtype, loop=True)
def exp(x, dtype=float32): return helper.looper(helper, x, call=math.exp, dtype=dtype, loop=True)
def exp2(x, dtype=int32): return helper.looper(helper, x, call=helper.exp2, dtype=dtype, loop=True)
def cbrt(x, dtype=int32): return helper.looper(helper, x, call=helper.cbrt, dtype=dtype, loop=True)
def sum(x, dtype=int32): return helper.looper(helper, x, call=builtins.sum, dtype=dtype, loop=True)
def mod(x, y, dtype=float32): return helper.looper(helper, x, call=helper.mod, y=y, dtype=dtype, loop=True)
def prod(x, y=0, dtype=int32): return helper.looper(helper, x, call=math.prod, dtype=dtype, loop=True)
def multiply(x, y=0, dtype=int32): return helper.looper(helper, x, call=math.prod, dtype=dtype, loop=True)
def zeros(s, d=0, dtype=float32): return helper.looper(helper, s, fill=d, noloop=True)
def zeros_like(s, d=0, dtype=int32): return empty(s, fill=0, like=True)
def ones(s, d=1, dtype=int32): return zeros(s, d) # call the same function as zeros but set it to 1 instead
def ones_like(s, d=1, dtype=int32): return empty(s, fill=1, like=True)
def max(x): return helper.looper(helper, x, max=True)
def min(x): return helper.looper(helper, x, min=True)
def maximum(x, y): return helper.looper(helper, x, y=y, maxi=True)
def minimum(x, y): return helper.looper(helper, x, y=y, mini=True)
def empty(x, fill=0, like=False): return helper.looper(helper, x, fill=fill, like=like, noloop=True)
def full(x, fill): return helper.looper(helper, x, fill=fill, noloop=True)
def cos(x, dtype=int32): return helper.looper(helper, x, call=math.cos, dtype=dtype, loop=True)
def sqrt(x, dtype=int32): return helper.looper(helper, x, call=math.sqrt, dtype=dtype, loop=True)
def arctan2(x, y, dtype=int32): return helper.looper(helper, x, call=math.atan2, y=y, dtype=dtype, loop=True)
def amax(x, dtype=int32): return helper.looper(helper, x, max=True) # Kinda works
def isin(x, y, dtype=int32): return helper.looper(helper, x, y=y, ret=[], isin=True)
def ceil(x, dtype=int32): return helper.looper(helper, x, call=math.ceil, dtype=dtype, loop=True)
def count_nonzero(x): return helper.looper(helper, x, ret=0, count=True)
def not_equal(x, y): return not testing.assert_equal(x, y)
def array_equal(x, y): return testing.assert_equal(x, y)
def arange(x,y,z, dtype=int32): return ndarray(helper.looper(helper, ret=0, fill=rnd.random(), noloop=True))
def any(x): return builtins.any(helper.looper(helper, x, ret=0, count=True))
def all(x): return builtins.all(helper.looper(helper, x, ret=0, count=True))
def copy(x): return helper.looper(helper, x, call=cp.copy, dtype=dtype, loop=True)
def median(x, r=[]):
  for i in range(len(x)): r.append(helper.looper(helper, x[i], add=True) // len(x[i]))
  return [r[i] / r[i + 1] for i in range(len(r) - 1)].pop()

def copyto(x, y):
  for i in range(len(y)): x[i] = copy(y)[i]
def set_printoptions(): pass
def allclose(x, y, rtol=1e-05, atol=1e-08):
  r = helper.looper(helper, x, y=y, ret=[], ass=True, asscls=True, cls=True)
  return not builtins.any((r[i] <= atol + rtol * r[i + 2]) is False for i in range(1, len(r), 4))

def reshape(l, shape): return helper.reshape(l, shape)
"""
    ncols, nrows, ret = 0, 0, []
    if shape == -1:
      if not isinstance(l, (list, tuple)): ncols, nrows = l, 1
      else: ncols, nrows = len(l), 1
    elif isinstance(shape, tuple): nrows, ncols = shape
    else: ncols, nrows = len(l), 1
    for r in range(nrows):
      row = []
      for c in range(ncols):
        if shape == -1 and isinstance(l, list) and not isinstance(l[c], (float, int)): row.extend(reshape(l[c], -1))
        elif shape == -1: row.extend(l); break
        else: row.append(l[ncols * r + c])
      if shape == -1: ret.extend(row)
      else: ret.append(row)
    return ret
"""
class ctypeslib: # kindof
  def as_array(x, shape): return arr.array('i', x)

class lib:
  class stride_tricks:
    def as_strided(self): pass

def concatenate(): pass
def expand_dims(): pass
def eye(): pass
def frombuffer(): pass
def stack(): pass
def tile(): pass
def argmax(): pass
def outer(): pass
def clip(): pass
def pad(): pass
def squeeze(): pass
def broadcast_to(): pass
def nonzero(): pass
def unique(): pass
def promote_types(): pass
def triu(): pass
def where(): pass
def dtype(): pass
def memmap(): pass
def require(): pass
def split(): pass
def moveaxis(): pass
def transpose(): pass
def rollaxis(): pass
def matmul(): pass
def argsort(): pass
def newaxis(): pass
def meshgrid(): pass
def delete(): pass
def vstack(): pass
def save(): pass
def load(): pass
def cumsum(): pass
def flip(): pass
