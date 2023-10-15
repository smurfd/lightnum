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
def prod(x, dtype=int32): return helper.looper_mul(x, dtype=dtype)
def multiply(x, dtype=int32): return helper.looper_mul(x, dtype=dtype)
def cos(x, dtype=int32): return helper.looper_cos(x, dtype=dtype)
def sqrt(x, dtype=int32): return helper.looper_sqrt(x, dtype=dtype)
def arctan2(x, y, dtype=int32): return helper.looper_atan2(x, y=y, dtype=dtype)
def ceil(x, dtype=int32): return helper.looper_ceil(x, dtype=dtype)
def copy(x, dtype=int32): return helper.looper_cp(x, dtype=dtype)

def empty(x, fill=0): return helper.looper_empty(x, fill=fill)
def full(x, fill): return helper.looper_empty(x, fill=fill)
def zeros(x, fill=0, dtype=float32): return helper.looper_empty(x, fill=fill)
def zeros_like(x, fill=0, dtype=int32): return helper.looper_empty_like(x, fill=0)
def ones(x, fill=1, dtype=int32): return zeros(x, fill)
def ones_like(x, fill=1, dtype=int32): return helper.looper_empty_like(x, fill=1)
def arange(start,stop=0,step=1, dtype=int32):
  if stop: return ndarray([i for i in range(start, stop, step)])
  return ndarray([i for i in range(0, start, step)])
def max(x): return helper.looper_max(x)
def min(x): return helper.looper_min(x)
def maximum(x, y): return helper.looper_maxi(x, y=y)
def minimum(x, y): return helper.looper_mini(x, y=y)
def amax(x, dtype=int32): return helper.looper_max(x)# Kinda works
def flip(x, dtype=int32): return helper.looper_flip(x)
def isin(x, y, dtype=int32): return helper.looper_isin(x, y=y, ret=[])
def count_nonzero(x): return helper.looper_count(x, ret=0)
def any(x): return builtins.any(helper.looper_count(x, ret=0))
def all(x): return builtins.all(helper.looper_count(x, ret=0))

def not_equal(x, y): return not testing.assert_equal(x, y)
def array_equal(x, y): return testing.assert_equal(x, y)
def reshape(l, shape): return helper.reshape(l, shape)
def median(x, r=[]):
  for i in range(len(x)): r.append(helper.looper_add(x[i]) // len(x[i]))
  return [r[i] / r[i + 1] for i in range(len(r) - 1)].pop()

def copyto(x, y):
  for i in range(len(y)): x[i] = copy(y)[i]
def set_printoptions(): pass
def allclose(x, y, rtol=1e-05, atol=1e-08):
  r = helper.looper_assert_close_cls(x, y)
  return not builtins.any((r[i] <= atol + rtol * r[i + 2]) is False for i in range(1, len(r), 4))

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
