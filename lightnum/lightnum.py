from lightnum.array import array, ndarray, asarray
from lightnum.random import random
from lightnum.testing import testing
from lightnum.helper import helper
from lightnum.dtypes import int32, float32
import random as rnd
import array as arr
import copy as cp
import builtins
import math
# TODO: make dtype=xxx do something
# TODO: class lib
# TODO: add functionallity to functions with pass

# math
def log(x, dtype=int32): return helper.looper_log(x, dtype=dtype)
def exp(x, dtype=float32): return helper.looper_exp(x, dtype=dtype)
def exp2(x, dtype=int32): return helper.looper_exp2(x, dtype=dtype)
def cbrt(x, dtype=int32): return helper.looper_cbrt(x, dtype=dtype)
def sum(x, dtype=int32): return helper.looper_sum(x, dtype=dtype)
def mod(x, y, dtype=float32): return helper.looper_mod(x, y=y, dtype=dtype)
def prod(x, dtype=int32): return helper.looper_prod(x, dtype=dtype)
def multiply(x, dtype=int32): return helper.looper_prod(x, dtype=dtype)
def cos(x, dtype=int32): return helper.looper_cos(x, dtype=dtype)
def sqrt(x, dtype=int32): return helper.looper_sqrt(x, dtype=dtype)
def arctan2(x, y, dtype=int32): return helper.looper_arctan2(x, y=y, dtype=dtype)
def ceil(x, dtype=int32): return helper.looper_ceil(x, dtype=dtype)
def copy(x, dtype=int32): return helper.looper_copy(x, dtype=dtype)
def matmul(x, y, dtype=float32): return helper.looper_matmul(x, y=y, dtype=dtype)
def empty(x, fill=0): return helper.looper_empty(x, fill=fill)
def full(x, fill): return helper.looper_empty(x, fill=fill)
def zeros(x, fill=0, dtype=float32): return helper.looper_empty(x, fill=fill)
def zeros_like(x, fill=0, dtype=int32): return helper.looper_empty_like(x, fill=0)
def ones(x, fill=1, dtype=int32): return zeros(x, fill)
def ones_like(x, fill=1, dtype=int32): return helper.looper_empty_like(x, fill=1)
def arange(start, stop=0, step=1, dtype=int32): return helper.looper_arange(start, stop, step, dtype)
def max(x): return helper.looper_max(x)
def min(x): return helper.looper_min(x)
def maximum(x, y): return helper.looper_maximum(x, y=y)
def minimum(x, y): return helper.looper_minimum(x, y=y)
def amax(x, dtype=int32): return helper.looper_max(x)# Kinda works
def flip(x, dtype=int32): return helper.looper_flip(x)
def split(x, y, dtype=int32): return helper.looper_split(x, y)
def tile(x, y, dtype=int32): return helper.looper_tile(x, y)
def isin(x, y, dtype=int32): return helper.looper_isin(x, y=y, ret=[])
def count_nonzero(x): return helper.looper_count(x, ret=0)
def broadcast_to(x, y): return helper.looper_broadcast_to(x, y)
def any(x): return builtins.any(helper.looper_count(x, ret=0))
def all(x): return builtins.all(helper.looper_count(x, ret=0))
def where(condition, x, y): return helper.looper_where(condition, x, y)
def not_equal(x, y): return not testing.assert_equal(x, y)
def array_equal(x, y): return testing.assert_equal(x, y)
def reshape(l, shape): return helper.reshape(l, shape)
def cumsum(x, dtype=int32): return helper.looper_cumsum(x)
def outer(x, y): x = reshape(x, -1); y = reshape(y, -1); return helper.looper_outer(x, y=y)
def expand_dims(x, axis): return helper.looper_expand_dims(x, axis)
def median(x, r=[]):
  for i in range(len(x)): r.append(helper.looper_add(x[i]) // len(x[i]))
  return [r[i] / r[i + 1] for i in range(len(r) - 1)].pop()
def concatenate(x): return helper.looper_concatenate(x)
def copyto(x, y):
  for i in range(len(y)): x[i] = copy(y)[i]
def set_printoptions(): pass
def allclose(x, y, rtol=1e-05, atol=1e-08):
  r = helper.looper_assert_close_cls(x, y)
  return not builtins.any((r[i] <= atol + rtol * r[i + 2]) is False for i in range(1, len(r), 4))
def eye(x, y=None, k=0): return [[1 if (xx-k)==yy else 0 for xx in range(y if y else x)] for yy in range(x)]

class ctypeslib: # kindof
  def as_array(x, shape): return arr.array('i', x)

class lib:
  class stride_tricks:
    def as_strided(self): pass

def frombuffer(): pass
def stack(): pass
def argmax(): pass
def clip(): pass
def pad(): pass
def squeeze(): pass
def nonzero(): pass
def unique(): pass
def promote_types(): pass
def triu(): pass
def dtype(): pass
def memmap(): pass
def require(): pass
def moveaxis(): pass
def transpose(): pass
def rollaxis(): pass
def argsort(): pass
def newaxis(): pass
def meshgrid(): pass
def delete(): pass
def vstack(): pass
def save(): pass
def load(): pass
