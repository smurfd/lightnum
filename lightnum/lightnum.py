from lightnum.array import array, ndarray, asarray
from lightnum.random import random
from lightnum.testing import testing
from lightnum.helper import helper
from lightnum.dtypes import int32, uint32, float16, float32, float64, uint8, dtype, types
import random as rnd
import array as arr
import copy as cp
import builtins
import math
import os
# TODO: make dtype=xxx do something
# TODO: add functionallity to functions with pass

# math
def log(x, dtype=float64): return helper.looper_log(x, dtype=dtype) if str(dtype) == types[dtype(x)] else helper.cast(helper.looper_log(x, dtype=dtype), dtype=dtype)
def log2(x, dtype=float64): return helper.looper_log(x, dtype=dtype) if str(dtype) == types[dtype(x)] else helper.cast(helper.looper_log2(x, dtype=dtype), dtype=dtype)
def exp(x, dtype=float32): return helper.looper_exp(x, dtype=dtype) if str(dtype) == types[dtype(x)] else helper.cast(helper.looper_exp(x, dtype=dtype), dtype=dtype)
def exp2(x, dtype=int32): return helper.looper_exp2(x, dtype=dtype) if str(dtype) == types[dtype(x)] else helper.cast(helper.looper_exp2(x, dtype=dtype), dtype=dtype)
def cbrt(x, dtype=int32): return helper.looper_cbrt(x, dtype=dtype) if str(dtype) == types[dtype(x)] else helper.cast(helper.looper_cbrt(x, dtype=dtype), dtype=dtype)
def sum(x, dtype=float16): return helper.looper_sum(x, dtype=dtype) if str(dtype) == types[dtype(x)] else helper.cast(helper.looper_sum(x, dtype=dtype), dtype=dtype)
def mod(x, y, dtype=float32): return helper.looper_mod(x, y=y, dtype=dtype) if str(dtype) == types[dtype(x)] else helper.cast(helper.looper_mod(x, y=y, dtype=dtype), dtype=dtype)
def add(x, y, dtype=float32): return helper.looper_add(x, y=y, dtype=dtype) if str(dtype) == types[dtype(x)] else helper.cast(helper.looper_add(x, y=y, dtype=dtype), dtype=dtype)
def subtract(x, y, dtype=float32): return helper.looper_subtract(x, y=y, dtype=dtype) if str(dtype) == types[dtype(x)] else helper.cast(helper.looper_subtract(x, y=y, dtype=dtype), dtype=dtype)
def prod(x, dtype=int32): return helper.looper_prod(x, dtype=dtype) if str(dtype) == types[dtype(x)] else helper.cast(helper.looper_prod(x, dtype=dtype), dtype=dtype)
def cos(x, dtype=int32): return helper.looper_cos(x, dtype=dtype) if str(dtype) == types[dtype(x)] else helper.cast(helper.looper_cos(x, dtype=dtype), dtype=dtype)
def ceil(x, dtype=int32): return helper.looper_ceil(x, dtype=dtype) if str(dtype) == types[dtype(x)] else helper.cast(helper.looper_ceil(x, dtype=dtype), dtype=dtype)
def copy(x, dtype=int32): return helper.looper_copy(x, dtype=dtype) if str(dtype) == types[dtype(x)] else helper.cast(helper.looper_copy(x, dtype=dtype), dtype=dtype)
def where(condition, x, y, dtype=int32): return helper.looper_where(condition, x, y, dtype=dtype) if str(dtype) == types[dtype(x)] else helper.cast(helper.looper_where(condition, x, y), dtype=dtype)
def sqrt(x, dtype=float32): return helper.looper_sqrt(x, dtype=dtype) if str(dtype) == types[dtype(x)] else helper.cast(helper.looper_sqrt(x, dtype=dtype), dtype=dtype)
def arctan2(x, y, dtype=float32): return helper.looper_arctan2(x, y=y, dtype=dtype) if str(dtype) == types[dtype(x)] else helper.cast(helper.looper_arctan2(x, y=y, dtype=dtype), dtype=dtype)
def multiply(x, dtype=int32): return helper.looper_prod(x, dtype=dtype) if str(dtype) == types[dtype(x)] else helper.cast(helper.looper_prod(x, dtype=dtype), dtype=dtype)
def matmul(x, y, dtype=float32): return helper.looper_matmul(x, y=y, dtype=dtype) if str(dtype) == types[dtype(x)] else helper.cast(helper.looper_matmul(x, y=y, dtype=dtype), dtype=dtype)
def empty(x, fill=0, dtype=int32): return helper.looper_empty(x, fill=fill, dtype=dtype) if str(dtype) == types[dtype(x)] else helper.cast(helper.looper_empty(x, fill=fill, dtype=dtype), dtype=dtype)
def full(x, fill, dtype=float64): return helper.looper_empty(x, fill=fill, dtype=dtype) if str(dtype) == types[dtype(x)] else helper.cast(helper.looper_empty(x, fill=fill, dtype=dtype), dtype=dtype)
def zeros(x, fill=0, dtype=float32): return helper.looper_empty(x, fill=fill, dtype=dtype) if str(dtype) == types[dtype(x)] else helper.cast(helper.looper_empty(x, fill=fill, dtype=dtype), dtype=dtype)
def zeros_like(x, fill=0, dtype=int32): return helper.looper_empty_like(x, fill=0, dtype=dtype) if str(dtype) == types[dtype(x)] else helper.cast(helper.looper_empty_like(x, fill=0, dtype=dtype), dtype=dtype)
def ones(x, fill=1, dtype=float64): return zeros(x, fill, dtype=dtype) if str(dtype) == types[dtype(x)] else helper.cast(zeros(x, fill, dtype=dtype), dtype=dtype)
def ones_like(x, fill=1, dtype=float32): return helper.looper_empty_like(x, fill=1, dtype=dtype) if str(dtype) == types[dtype(x)] else helper.cast(helper.looper_empty_like(x, fill=1, dtype=dtype), dtype=dtype)
def max(x): return helper.looper_max(x)
def min(x): return helper.looper_min(x)
def maximum(x, y): return helper.looper_maximum(x, y=y)
def minimum(x, y): return helper.looper_minimum(x, y=y)
def amax(x, dtype=int32): return helper.looper_max(x) if str(dtype) == types[dtype(x)] else helper.cast(helper.looper_max(x), dtype=dtype)# Kinda works
def flip(x, dtype=int32): return helper.looper_flip(x, dtype=dtype) if str(dtype) == types[dtype(x)] else helper.cast(helper.looper_flip(x, dtype=dtype), dtype=dtype)
def split(x, y, dtype=int32): return helper.looper_split(x, y, dtype=dtype) if str(dtype) == types[dtype(x)] else helper.cast(helper.looper_split(x, y, dtype=dtype), dtype=dtype)
def tile(x, y, dtype=int32): return helper.looper_tile(x, y, dtype=dtype) if str(dtype) == types[dtype(x)] else helper.cast(helper.looper_tile(x, y, dtype=dtype), dtype=dtype)
def isin(x, y, dtype=int32): return helper.looper_isin(x, y=y, ret=[]) if str(dtype) == types[dtype(x)] else helper.cast(helper.looper_isin(x, y=y, ret=[]), dtype=dtype)
def count_nonzero(x): return helper.looper_count(x, ret=0)
def broadcast_to(x, y): return helper.looper_broadcast_to(x, y)
def any(x): return builtins.any(helper.looper_count(x, ret=0))
def all(x): return builtins.all(helper.looper_count(x, ret=0))
def not_equal(x, y): return not testing.assert_equal(x, y)
def array_equal(x, y): return testing.assert_equal(x, y)
def reshape(l, shape): return helper.reshape(l, shape)
def cumsum(x, dtype=int32): return helper.looper_cumsum(x, dtype=dtype) if str(dtype) == types[dtype(x)] else helper.cast(helper.looper_cumsum(x, dtype=dtype), dtype=dtype)
def outer(x, y): x = reshape(x, -1); y = reshape(y, -1); return helper.looper_outer(x, y=y)
def expand_dims(x, axis): return helper.looper_expand_dims(x, axis)
def argmax(x, axis=None): return helper.looper_argmax(x, axis)
def transpose(x, axes=None): return helper.looper_transpose(x, axes)
def stack(x, axis=0): return helper.looper_stack(x, axis)
def squeeze(x, axis=0): return helper.looper_squeeze(x, axis)
def clip(x, x_min, x_max): return helper.looper_clip(x, x_min, x_max)
def unique(x): return helper.looper_unique(x)
def vstack(x): return helper.looper_vstack(x)
def nonzero(x): return helper.looper_nonzero(x)
def less(x, y): return x < y
def equal(x, y): return x == y
def promote_types(x, y): return dtype(x) if dtype(x) <= dtype(y) else dtype(y)
def median(x):
  r = []
  for i in range(len(x)): r.append(helper.looper_median(x[i]) // len(x[i]))
  return [r[i] / r[i + 1] for i in range(len(r) - 1)].pop()
def concatenate(x): return helper.looper_concatenate(x)
def copyto(x, y):
  for i in range(len(y)): x[i] = copy(y)[i]
def set_printoptions(): pass
def allclose(x, y, rtol=1e-05, atol=1e-08):
  r = helper.looper_assert_close_cls(x, y)
  return not builtins.any((r[i] <= atol + rtol * r[i + 2]) is False for i in range(1, len(r), 4))
def eye(x, y=None, k=0): return [[1 if (xx-k)==yy else 0 for xx in range(y if y else x)] for yy in range(x)]
def frombuffer(buf, dtype=int32): return helper.looper_frombuffer(buf, dtype)
def sin(x, dtype=float64): return helper.looper_sin(x, dtype=dtype) if str(dtype) == types[dtype(x)] else helper.cast(helper.looper_sin(x, dtype=dtype), dtype=dtype)
def reciprocal(x, dtype=float64): return helper.looper_reciprocal(x, dtype=dtype) if str(dtype) == types[dtype(x)] else helper.cast(helper.looper_reciprocal(x, dtype=dtype), dtype=dtype)
class arange():
  def __init__(self, start, stop=0, step=1, dtype=int32): self.ret = helper.looper_arange(start, stop, step, dtype)
  def __call__(self): return self.ret
  def reshape(self, s): return reshape(self.ret, s)
  def tolist(self): return self.ret
def triu(x, l=0):
  l = l - 1
  if isinstance(x[0], list):
    rr = []
    if (l + 1) >= len(x[0]): l = len(x[0]) - 1
    for i in range(len(x)):
      rr.append(helper.zero_row_len(x[i], l:=l+1))
    return rr
  else:
    r1, rr = [], []
    for i in range(len(x)): r1.append(x)
    for i in range(len(r1)):
      f = []
      f.extend(helper.zero_row_len(r1[i], l:=l+1))
      rr.append(f)
    return rr

def newaxis(x, y): # if y is set, it represents [np.newaxis, :], if not [:, np.newaxis]
  ret = []
  if y:
    if not isinstance(x[0], list): ret = [x]
    else:
      for i in range(len(x)): ret += [x[i]]
      ret = [ret]
  else:
    for i in range(len(x)): ret.append([x[i]])
  return ret

def meshgrid(x, y, indexing='xy'):
  retx, rety = [], []
  rx = reshape(x, -1)
  ry = reshape(y, -1)
  if indexing == 'xy':
    for i in range(len(rx)): retx.append(rx)
    for i in range(len(ry)): rety.append(full(ry, fill=ry[i]))
  elif indexing == 'ij':
    for i in range(len(rx)): retx.append(full(rx, fill=rx[i]))
    for i in range(len(ry)): rety.append(ry)
  else: pass
  return retx, rety

def load(f):
  ret = []
  if isinstance(f, str) and f.endswith('npz'):
    from zipfile import ZipFile
    with open(f, 'rb') as ff:
      with ZipFile(ff) as myzipfile:
        for x in myzipfile.filelist: ret.append((os.path.splitext(x.filename)[0], load(myzipfile.open(x.filename))))
        return dict(ret)
  elif isinstance(f, str):
    with open(f, 'rb') as ff: ret.append(load(ff))
  else:
    if not helper.read_magic(f): raise TypeError("Not a magic file")
    d1, d2 = helper.read_header(f, helper.read_header_type(f))
    if len(d1) == 1: ret = helper.read_body(f, (d1[0])*8)
    else: ret.extend([helper.read_body(f, (d1[len(d1)-1])*8) for _ in range(d1[0])])
  return ret

def save(f, x):
  hdr = "{{\'descr\': \'<i8\', \'fortran_order\': False, \'shape\': ({},), }}".format(len(x))
  helper.write_magic(f)
  helper.write_header_type(f, (1, 0))
  helper.write_header_len(f, (1, 0), hdr.ljust(118))
  helper.write_header(f, (1, 0), hdr.ljust(118))
  helper.write_body(f, x)

def delete(x, y, axis=None): # kindof
  if isinstance(y, int):
    if axis is None:
      z = reshape(x, -1)
      return z[:y] + z[y + 1:]
    return x[:y] + x[y + 1:]

# TODO : needs to work on +3D arrays
def pad(x, y, mode='constant', **kwargs): return helper.looper_pad(x, y, mode, **kwargs)

class ctypeslib: # kindof
  def as_array(x, shape): return arr.array('i', x)

class lib:
  class stride_tricks:
    def as_strided(self): pass

def memmap(): pass
def require(): pass
def moveaxis(): pass
def rollaxis(): pass
def argsort(): pass
