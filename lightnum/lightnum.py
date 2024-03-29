#!/usr/bin/env python3
from lightnum.array import ndarray
from lightnum.testing import testing as testing
from lightnum.array import array as array
from lightnum.helper import helper
from lightnum.random import random as random
from lightnum.dtypes import int32 as int32, float16 as float16, float32 as float32, float64 as float64, dtype as dtype, types as types, uint8 as uint8, uint32 as uint32
import array as arr, copy as cp, builtins, ctypes, math
from typing import List, Any, Type, BinaryIO, Tuple, Union, SupportsIndex, SupportsFloat, Iterable

# TODO: Fix all kindofs
# TODO: Compare arguments to functions towards numpy functions
# TODO: Remove more helpers
# TODO: See if speedups can be done
# TODO: Integrate with tinygrad
h = helper()
t = testing()
def cast(x: List[Any], dtype: dtype = float64) -> List[Any]:
  if isinstance(x, list): return list([cast(i, dtype) for i in x])
  a, b = ctypes.cast(ctypes.pointer(dtype(x)(round(x, 8))), ctypes.POINTER(dtype(x))).contents.value, (not isinstance(x, list) and x != float(int(x)))
  return (round(a, 8 if len(str(x)) > 8 else len(str(x))) if b else int(a) if dtype in [float16, float32, float64] else a)

# math
def log(x: Union[Any, List[Any]], dtype: dtype = float64) -> Union[Any, List[Any]]: return round(math.log(x),8) if not isinstance(x, list) else [log(i) for i in x] if repr(dtype) == types[dtype(x)] else cast(log(x), dtype=dtype)
def log2(x: Union[SupportsFloat, SupportsIndex], dtype: dtype = float64) -> Union[Any, List[Any]]: return round(math.log2(x),8) if not isinstance(x, list) else [log2(i) for i in x] if repr(dtype) == types[dtype(x)] else cast(log2(x), dtype=dtype)
def exp(x: Union[Any, List[Any]], dtype: dtype = float64) -> Union[Any, List[Any]]: return math.exp(x) if not isinstance(x, list) else [exp(i) for i in x] if repr(dtype) == types[dtype(x)] else cast(exp(x), dtype=dtype)
def exp2(x: Union[Any, List[Any]], dtype: dtype = float32) -> Union[Any, List[Any]]: return 2 ** x if not isinstance(x, list) else [exp2(i) for i in x] if repr(dtype) == types[dtype(x)] else cast(exp2(x), dtype=dtype)
def cbrt(x: Union[Any, List[Any]], dtype: dtype = float32) -> Union[Any, List[Any]]: return h.cbrt(x) if not isinstance(x, list) else [cbrt(i) for i in x] if repr(dtype) == types[dtype(x)] else cast(cbrt(x), dtype=dtype)
def sum(x: Union[Any, List[Any]], dtype: dtype = float16) -> Union[Any, List[Any]]: return builtins.sum(x) if not isinstance(x, list) else [sum(i) for i in x] if repr(dtype) == types[dtype(x)] else cast(sum(x), dtype=dtype)
def sqrt(x: Union[Any, List[Any]], dtype: dtype = float64) -> Union[Any, List[Any]]: return math.sqrt(x) if not isinstance(x, list) else [sqrt(i) for i in x] if repr(dtype) == types[dtype(x)] else cast(sqrt(x), dtype=dtype)
def mod(x: Union[Any, List[Any]], y: Union[Any, List[Any]], dtype: dtype = float32) -> Union[Any, List[Any]]: return (x%y) if not isinstance(x, (list, tuple)) else [mod(i, y=j) for i, j in zip(x, y)] if repr(dtype) == types[dtype(x)] else cast(mod(x, y), dtype=dtype)
def add(x: Union[Any, List[Any]], y: Union[Any, List[Any]], dtype: dtype = float32) -> Union[Any, List[Any]]: return (x+y) if not isinstance(x, (list, tuple)) else [add(i, y=j) for i, j in zip(x, y)] if repr(dtype) == types[dtype(x)] else cast(add(x, y), dtype=dtype)
def arctan2(x: Union[SupportsFloat, SupportsIndex, List[Any]], y: Union[SupportsFloat, SupportsIndex, List[Any]], dtype: dtype = float32) -> Union[Any, List[Any]]: return math.atan2(x, y) if not isinstance(x, (list, tuple)) else [arctan2(i, y=j) for i, j in zip(list(x), list(y))] if repr(dtype) == types[dtype(x)] else cast(arctan2(x, y), dtype=dtype) #type: ignore
def prod(x: Union[Any, List[Any]], dtype: dtype = float64) -> Union[Any, List[Any]]: return math.prod(x) if not isinstance(x, list) else [prod(i) for i in x] if repr(dtype) == types[dtype(x)] else cast(prod(x), dtype=dtype)
def cos(x: Union[Any, List[Any]], dtype: dtype = float64) -> Union[Any, List[Any]]: return math.cos(x) if not isinstance(x, list) else [cos(i) for i in x] if repr(dtype) == types[dtype(x)] else cast(cos(x), dtype=dtype)
def ceil(x: Union[Any, List[Any]], dtype: dtype = float64) -> Union[Any, List[Any]]: return math.ceil(x) if not isinstance(x, list) else [ceil(i) for i in x] if repr(dtype) == types[dtype(x)] else cast(ceil(x), dtype=dtype)
def sin(x: Union[Any, List[Any]], dtype: dtype = float64) -> Union[Any, List[Any]]: return math.sin(x) if not isinstance(x, list) else [sin(i) for i in x] if repr(dtype) == types[dtype(x)] else cast(sin(x), dtype=dtype)
def subtract(x: Union[Any, List[Any]], y: Union[Any, List[Any]], dtype: dtype = float32) -> Union[Any, List[Any]]: return (x-y) if not isinstance(x, (list, tuple)) else [subtract(i, y=j) for i, j in zip(x, y)] if repr(dtype) == types[dtype(x)] else cast(subtract(x, y), dtype=dtype)
def copy(x: Union[Any, List[Any]], dtype: dtype = float64) -> Union[Any, List[Any]]: return cp.copy(x) if not isinstance(x, list) else [copy(i) for i in x] if repr(dtype) == types[dtype(x)] else cast(copy(x), dtype=dtype)
def multiply(x: Union[Any, List[Any]], dtype: dtype = float64) -> Union[Any, List[Any]]: return math.prod(x) if not isinstance(x, list) else [multiply(i) for i in x] if repr(dtype) == types[dtype(x)] else cast(multiply(x), dtype=dtype)
def reciprocal(x: Union[Any, List[Any]], dtype: dtype = float64) -> Union[Any, List[Any]]: return 1/(x) if not isinstance(x, list) else [reciprocal(i) for i in x] if repr(dtype) == types[dtype(x)] else cast(reciprocal(x), dtype=dtype)
def frombuffer(x: Union[Any, List[Any]], dtype: dtype = float64) -> Union[Any, List[Any]]: return [i for i in x] if not isinstance(x, list) else [frombuffer(i) for i in x] if repr(dtype) == types[dtype(x)] else cast(frombuffer(x), dtype=dtype)
def where(condition: str, x: Union[Any, List[Any]], y: Union[Any, List[Any]], dtype: dtype = int32) -> Union[Any, List[Any]]: return [xv if c else yv for c, xv, yv in zip(condition, x, y)] if not isinstance(x, (list, tuple)) else [where(condition, i, y=j) for i, j in zip(x, y)] if repr(dtype) == types[dtype(x)] else cast(where(condition, x, y), dtype=dtype)
def argsort(x: Union[Any, List[Any]]) -> Union[Any, List[Any]]:
  if not isinstance(x[0], list): return sorted(range(len(x)), key=x.__getitem__)
  else: return [argsort(x[i]) for i in range(len(x))]
def matmul(x: List[Any], y: List[Any], dtype: dtype = float32) -> Any:
  if str(dtype) == types[dtype(x)]:
    if not isinstance(x, (list, tuple)):
      if x and y: return x * y
      elif x and not y: return x
      else: return y
    return reshape([matmul(i, y=j) for i, j in zip(x, y)], -1)
  else: h.cast(matmul(x, y=y, dtype=dtype), dtype=dtype)

def empty(x: Union[Any, List[Any]], fill: Union[int, float] = 0, dtype: dtype = int32) -> Any: return h.looper_empty(x, fill=fill, dtype=dtype) if str(dtype) == types[dtype(x)] else h.cast(h.looper_empty(x, fill=fill, dtype=dtype), dtype=dtype)
def full(x: Union[Any, List[Any]], fill: Union[int, float], dtype: dtype = float64) -> Any: return h.looper_empty(x, fill=fill, dtype=dtype) if str(dtype) == types[dtype(x)] else h.cast(h.looper_empty(x, fill=fill, dtype=dtype), dtype=dtype)
def zeros(x: List[Any], fill: Union[int, float] = 0, dtype: dtype = float32) -> Any: return h.looper_empty(x, fill=fill, dtype=dtype) if str(dtype) == types[dtype(x)] else h.cast(h.looper_empty(x, fill=fill, dtype=dtype), dtype=dtype)
def zeros_like(x: List[Any], fill: Union[int, float] = 0, dtype: dtype = int32) -> Any: return h.looper_empty_like(x, fill=0, dtype=dtype) if str(dtype) == types[dtype(x)] else h.cast(h.looper_empty_like(x, fill=0, dtype=dtype), dtype=dtype)
def ones(x: List[Any], fill: Union[int, float] = 1, dtype: dtype = float64) -> Any: return h.looper_empty(x, fill, dtype=dtype) if str(dtype) == types[dtype(x)] else h.cast(zeros(x, fill, dtype=dtype), dtype=dtype)
def ones_like(x: List[Any], fill: Union[int, float] = 1, dtype: dtype = float32) -> Any: return h.looper_empty_like(x, fill=1, dtype=dtype) if str(dtype) == types[dtype(x)] else h.cast(h.looper_empty_like(x, fill=1, dtype=dtype), dtype=dtype)
def max(x: List[Any]) -> Union[float, int, List[Any]]: return h.looper_max(x)
def min(x: List[Any]) -> Union[float, int, List[Any]]: return h.looper_min(x)
def maximum(x: List[Any], y: List[Any]) -> List[Any]: return h.looper_maximum(x, y=y)
def minimum(x: List[Any], y: List[Any]) -> List[Any]: return h.looper_minimum(x, y=y)
def amax(x: List[Any], dtype: dtype = int32) -> Any: return h.looper_max(x) if str(dtype) == types[dtype(x)] else h.cast(h.looper_max(x), dtype=dtype)# Kinda works
def flip(x: List[Any], dtype: dtype = int32) -> Any: return h.looper_flip(x, dtype=dtype) if str(dtype) == types[dtype(x)] else h.cast(h.looper_flip(x, dtype=dtype), dtype=dtype)
def split(x: Union[Any, List[Any]], y: Union[Any, List[Any]], dtype: dtype = int32) -> Any: return h.looper_split(x, y, dtype=dtype) if str(dtype) == types[dtype(x)] else h.cast(h.looper_split(x, y, dtype=dtype), dtype=dtype)
def tile(x: Union[Any, List[Any]], y: Union[Any, List[Any]], dtype: dtype = int32) -> Any: return h.looper_tile(x, y, dtype=dtype) if str(dtype) == types[dtype(x)] else h.cast(h.looper_tile(x, y, dtype=dtype), dtype=dtype)
def isin(x: List[Any], y: List[Any], dtype: dtype = int32) -> Any: return h.looper_isin(x, y=y) if str(dtype) == types[dtype(x)] else h.cast(h.looper_isin(x, y=y), dtype=dtype)
def count_nonzero(x: List[Any]) -> Union[int, Iterable[object]]: return h.looper_count(x)
def broadcast_to(x: List[Any], y: Union[List[Any], Tuple[int, int]]) -> List[Any]: return h.looper_broadcast_to(x, y)
def any(x: Union[int, Iterable[object], List[Any]]) -> bool: return builtins.any(h.looper_count(x)) #type: ignore
def all(x: Union[int, Iterable[object], List[Any]]) -> bool: return builtins.all(h.looper_count(x)) #type: ignore
def not_equal(x: List[Any], y: List[Any]) -> None: t.assert_equal(x, y)
def array_equal(x: List[Any], y: List[Any]) -> Any: return t.assert_equal(x, y)
def cumsum(x: List[Any], dtype: dtype = int32) -> Any: return h.looper_cumsum(x, dtype=dtype) if str(dtype) == types[dtype(x)] else h.cast(h.looper_cumsum(x, dtype=dtype), dtype=dtype)
def outer(x: List[Any], y: List[Any]) -> List[Any]: return [[x1*y1 for y1 in reshape(y, -1)] for x1 in reshape(x, -1)]
def expand_dims(x: List[Any], axis: Union[Any, int]) -> List[Any]: return h.looper_expand_dims(x, axis)
def argmax(x: List[Any], axis: Union[Any, int] = None) -> Union[int, List[Any]]: return h.looper_argmax(x, axis)
def transpose(x: List[Any], axes: Union[Any, int] = None) -> List[Any]: return h.looper_transpose(x, axes)
def stack(x: List[Any], axis: Union[Any, int] = 0) -> List[Any]: return h.looper_stack(x, axis)
def squeeze(x: List[Any], axis: Union[Any, int] = 0) -> List[Any]: return h.looper_squeeze(x, axis)
def clip(x: List[Any], x_min: int, x_max: int) -> Any: return h.looper_clip(x, x_min, x_max)
def unique(x: List[Any]) -> List[Any]: return list(set(sorted(h.reshape(x, -1))))
def vstack(x: List[Any]) -> List[Any]: return [i for i in x]
def nonzero(x: List[Any]) -> Tuple[ndarray, ...]: return h.looper_nonzero(x)
def less(x: List[Any], y: List[Any]) -> bool: return x < y
def equal(x: List[Any], y: List[Any]) -> bool: return x == y
def promote_types(x: Union[dtype, Type[Any]], y: Union[dtype, Type[Any]]) -> dtype: return dtype(x) if dtype(x) <= dtype(y) else dtype(y) #type: ignore
def median(x: List[Any]) -> float:
  r = []
  for i in range(len(x)): r.append(h.looper_median(x[i]) // len(x[i]))
  return [r[i] / r[i + 1] for i in range(len(r) - 1)].pop()
def concatenate(x: Union[Any, List[Any]]) -> List[Any]: return [i for j in range(len(x)) for i in x[j]]
def copyto(x: List[Any], y: List[Any]) -> None:
  for i in range(len(y)): x[i] = copy(y)[i]
def set_printoptions() -> None: pass
def allclose(x: List[Any], y: List[Any], rtol: float = 1e-05, atol: float = 1e-08) -> bool:
  for i in range(len(x)): r = h.if_round_abs(x[i], y[i], cls=True)
  return not builtins.any((r[i] <= atol + rtol * r[i + 2]) is False for i in range(1, len(r), 4))
def eye(x: int, y: int = 0, k: int=0) -> List[Any]: return [[1 if (xx-k)==yy else 0 for xx in range(y if y else x)] for yy in range(x)]
def moveaxis(x: List[Any], src: Union[int, List[Any]], dst: Union[int, List[Any]]) -> List[Any]:
  # kindof
  if src == -1 or dst == -1 or src == dst: return x
  return [[x[i][j] for i in range(len(x))] for j in range(len(x[0]))]
def rollaxis(x: List[Any], axis: int, start: int = 0) -> List[Any]:
  # kindof
  return moveaxis(x, start, axis)
def require(x: List[Any], dtype: dtype = int32, requirements: List[str] = ['C']) -> ndarray:
  ret = ndarray(x)
  for i in requirements:
    if i == 'A' or i == 'ALIGNED': ret.flags['ALIGNED'] = True
    elif i == 'W' or i == 'WRITEABLE': ret.flags['WRITEABLE'] = True
    elif i == 'O' or i == 'OWNDATA': ret.flags['OWNDATA'] = True
    elif i == 'E' or i == 'ENSUREARRAY': ret.flags['ENSUREARRAY'] = True
    elif i == 'C' or i == 'C_CONTIGUOUS': ret.flags['C_CONTIGUOUS'] = True
    elif i == 'F' or i == 'F_CONTIGUOUS': ret.flags['F_CONTIGUOUS'] = True
  return ret
def reshape(ar: Union[List[Any], ndarray], n: Union[int, List[Any], Tuple[int, int]], ret: Union[List[Any], Any] = None) -> List[Any]:
  if n == -1:
    if not ret: ret = []
    for i in list(ar):
      if isinstance(i, list): ret = reshape(i, -1, ret)
      else: ret.append(i)
    return ret
  l: int = int(str(n)) if not isinstance(n, tuple) else int(str(n[len(n) -1]))
  if len(list(ar)) % l or len(n) <= 2: #type: ignore
    return [ar[i: i + l] + [None] * (i + l - len(list(ar))) for i in range(0, len(list(ar)), l)] #type: ignore
  return [(i + l - len(ar)) for i in range(0, len(ar), l)]

class arange():
  def __init__(self, start: int, stop: int = 0, step: int = 1, dtype: dtype = int32) -> None:
    self.ret = h.looper_arange(start, stop, step, dtype)
    self.x: List[Any] = []
  def __call__(self) -> Union[List[Any], ndarray]: return self.ret
  def __mul__(self, x: List[Any]) -> List[Any]: return [i * x for i in self.x]
  def reshape(self, s: Union[int, List[Any]]) -> List[Any]: return reshape(self.ret, s)
  def tolist(self) -> Union[List[Any], ndarray]: return self.ret

def triu(x: List[Any], l: int = -1) -> Union[List[Any], ndarray]:
  if isinstance(x[0], list):
    rr = []
    if (l + 1) >= len(x[0]): l = len(x[0]) - 1
    for i in range(len(x)):
      rr.append(h.zero_row_len(x[i], l:=l+1))
    return rr
  else:
    r1, rr = [], []
    for i in range(len(x)): r1.append(x)
    for i in range(len(r1)):
      f = []
      f.extend(h.zero_row_len(r1[i], l:=l+1))
      rr.append(f)
    return rr

def newaxis(x: List[Any], y: Union[int, List[Any]]) -> List[Any]: # if y is set, it represents [np.newaxis, :], if not [:, np.newaxis]
  ret = []
  if y:
    if not isinstance(x[0], list): ret = [x]
    else:
      for i in range(len(x)): ret += [x[i]]
      ret = [ret]
  else:
    for i in range(len(x)): ret.append([x[i]])
  return ret

def meshgrid(x: List[Any], y: List[Any], indexing: str = 'xy') -> Tuple[List[Any], List[Any]]:
  retx, rety = [], []
  rx = reshape(x, -1)
  ry = reshape(y, -1)
  if indexing == 'xy':
    for i in range(len(rx)): retx.append(rx)
    for i in range(len(ry)): rety.append([ry[i]]*len(ry))
  elif indexing == 'ij':
    for i in range(len(rx)): retx.append([rx[i]]*len(rx))
    for i in range(len(ry)): rety.append(ry)
  return retx, rety

def load(f: BinaryIO) -> List[Any]:
  import os
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
    if not h.read_magic(f): raise TypeError("Not a magic file")
    d1, d2 = h.read_header(f, h.read_header_type(f))
    if len(d1) == 1: ret = h.read_body(f, (d1[0])*8)
    else: ret.extend((h.read_body(f, (d1[len(d1)-1])*8)) for _ in range(d1[0])) #type: ignore
  return ret

def save(f: BinaryIO, x: List[Any]) -> None:
  hdr = "{{\'descr\': \'<i8\', \'fortran_order\': False, \'shape\': ({},), }}".format(len(x))
  h.write_magic(f)
  h.write_header_type(f, (1, 0))
  h.write_header_len(f, (1, 0), hdr.ljust(118))
  h.write_header(f, (1, 0), hdr.ljust(118))
  h.write_body(f, x)

# kindof
def delete(x: List[Any], y: Union[int, List[Any]], axis: Union[int, Any] = None) -> List[Any]:
  if isinstance(y, int):
    if axis is None:
      z = reshape(x, -1)
      return z[:y] + z[y + 1:]
    return x[:y] + x[y + 1:]
  return []
# TODO : needs to work on +3D arrays
def pad(x: Union[Any, List[Any]], y: Union[Any, List[Any]], mode: str='constant', **kwargs: Any) -> List[Any]: return h.looper_pad(x, y, mode, **kwargs)

# kindof
class ctypeslib:
  def as_array(self, x: List[Any], shape: List[Any]) -> Union[List[Any], Type[Any]]: return arr.array('i', x) #type: ignore

class lib:
  class stride_tricks:
    def as_strided(self) -> None: pass
