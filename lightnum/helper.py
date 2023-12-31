from lightnum.dtypes import int16, int32, uint32, float16, float32, float64, uint8, uint16, types
from lightnum.array import ndarray, array
import copy as cp
import builtins
import ctypes
import struct
import math
import ast

class helper():
  MAGIC_PREFIX = b'\x93NUMPY'
  _header_size_info = {
    (1, 0): ('<H', 'latin1'),
    (2, 0): ('<I', 'latin1'),
    (3, 0): ('<I', 'utf8'),
  }
  def typ(x, dtype=int32): return dtype(x).value if not isinstance(x, int) else x
  def sum(x, y): return x + y
  def mod(x, y): return x % y
  def exp2(x): return 2 ** x
  def cbrt(x): return round(x**(1 / 3.), 2)
  def getrow(self, x, fill=0): return [helper.looper_getrow(x[-1], fill=fill) for _ in range(x[len(x) - 2])]
  def format_float(x): return float(('%i' if x == int(x) else '%s') % x)
  def cast(x, dtype=float64): return helper.looper_cast(x, dtype=dtype) if str(dtype) != types[dtype(x)] else x
  def zero_row_len(x, l):
    for i in range(l):
      if i >= 0 and i < len(x): x[i] = 0
    return x

  # helper functions to loop through multidimentional lists/tuples
  def looper_cast(x, dtype=float64):
    if isinstance(x, list): return [helper.looper_cast(i, dtype) for i in x]
    a = ctypes.cast(ctypes.pointer(dtype(x)(round(x, 8))), ctypes.POINTER(dtype(x))).contents.value
    b = (not isinstance(x, list) and x != float(int(x)))
    return (round(a, 8 if len(str(x)) > 8 else len(str(x))) if b else int(a) if dtype in [float16, float32, float64] else a)
  def looper_log(x, dtype=float64): return math.log(x) if not isinstance(x, list) else [helper.looper_log(i) for i in x]
  def looper_exp(x, dtype=float64): return math.exp(x) if not isinstance(x, list) else [helper.looper_exp(i) for i in x]
  def looper_exp2(x, dtype=float32): return helper.exp2(x) if not isinstance(x, list) else [helper.looper_exp2(i) for i in x]
  def looper_cbrt(x, dtype=float32): return helper.cbrt(x) if not isinstance(x, list) else [helper.looper_cbrt(i) for i in x]
  def looper_sum(x, dtype=float16): return builtins.sum(x) if not isinstance(x, list) else [helper.looper_sum(i) for i in x]
  def looper_mod(x, y, dtype=float32): return helper.mod(x, y) if not isinstance(x, (list, tuple)) else [helper.looper_mod(i, y=j) for i, j in zip(x, y)]
  def looper_prod(x, dtype=int32): return math.prod(x) if not isinstance(x, list) else [helper.looper_prod(i) for i in x]
  def looper_cos(x, dtype=float64): return math.cos(x) if not isinstance(x, list) else [helper.looper_cos(i) for i in x]
  def looper_ceil(x, dtype=float32): return math.ceil(x) if not isinstance(x, list) else [helper.looper_ceil(i) for i in x]
  def looper_copy(x, dtype=float32): return cp.copy(x) if not isinstance(x, list) else [helper.looper_copy(i) for i in x]
  def looper_where(condition, x, y, dtype=float32): return [xv if c else yv for c, xv, yv in zip(condition, x.tolist(), y)] if not isinstance(x, list) else [helper.looper_where(condition, i, j) for i,j in zip(x, y)]
  def looper_sqrt(x, dtype=float32): return math.sqrt(x) if not isinstance(x, list) else [helper.looper_sqrt(i) for i in x]
  def looper_arctan2(x, y, dtype=float32): return math.atan2(x, y) if not isinstance(x, (list, tuple)) else [helper.looper_arctan2(i, y=j) for i, j in zip(x, y)]
  def looper_transpose(x, axes=None): return [[row[i] for row in x] for i in range(len(x[0]))] #TODO: axes
  def looper_stack(x, axis=0): return [i for i in x] #TODO: axis
  def looper_vstack(x): return [i for i in x]
  def looper_unique(x): return list(set(sorted(helper.reshape(x, -1))))
  def looper_concatenate(x, dtype=int32): return [i for j in range(len(x)) for i in x[j]]
  def looper_expand_dims(x, axis, axisco=0): return [x] if axisco == axis else [helper.looper_expand_dims(x[i], axis, axisco + 1) for i in range(len(x))]
  def looper_flip(x, dtype=int32): return [helper.looper_flip(i) for i in x] if isinstance(x[0], list) else x[::-1]
  def looper_squeeze(x, axis=0): return x if not isinstance(x[0][0], list) else [helper.looper_squeeze(y, axis) for y in x].pop() #TODO: axis
  def looper_clip(x, x_min, x_max): return (x_min if x < x_min else x_max) if not isinstance(x, list) else [helper.looper_clip(y, x_min, x_max) for y in x]
  def looper_broadcast_to(x, y): return x if len(y) == 1 else [x for i in range(y[0])]
  def looper_outer(x, y): return [[x1*y1 for y1 in y] for x1 in x]
  def looper_frombuffer(buf, dtype=int32): return [helper.typ(i, dtype=dtype) for i in buf]
  def looper_nonzero(x):
    ret1, ret2 = [], []
    if isinstance(x[0], list): [[(ret1.append(i), ret2.append(ii)) for ii in range(len(x[i])) if x[i][ii]] for i in range(len(x)) if isinstance(x[i], list)]
    return tuple([ndarray(ret1)] + [ndarray(ret2)])

  def looper_arange(start, stop=0, step=1, dtype=int32):
    if stop: return ndarray([i for i in range(start, stop, step)])
    return ndarray([i for i in range(0, start, step)])

  def looper_argmax(x, axis=None):
    if not axis:
      y = helper.reshape(x, -1)
      return y.index(max(y))
    return x.index(max(x)) if not isinstance(x[0], list) else [helper.looper_argmax(y, axis) for y in x]

  def looper_cumsum(x, s=0,dtype=int32):
    a = helper.reshape(x, -1)
    return [builtins.sum(a[0:i:1]) for i in range(0, len(a)+1)][1:]

  def looper_add(x, dtype=int32, ret=0):
    for i in range(len(x)):
      if isinstance(x[i], (list, tuple)): ret = helper.looper_add(x[i], ret)
      else: ret += x[i]
    return ret

  def looper_split(x, y, dtype=int32):
    if not isinstance(y, list) and not isinstance(x, list): return [ndarray(x.tolist()[i:i+len(x.tolist())//y]) for i in range(0, len(x.tolist()), len(x.tolist())//y)]
    elif not isinstance(y, list): return [ndarray(x[i:i+(len(x)//y)]) for i in range(0, len(x), len(x)//y)]
    return [helper.looper_split(i, y) for i in x]

  def looper_tile(x, y, dtype=int32):
    tmp, tmp2=[], []
    if not isinstance(y, tuple):
      for _ in range(y): tmp.extend(x)
      return ndarray(tmp)
    a,b = y
    for a1 in range(a):
      for b1 in range(b): tmp2.extend(x)
      tmp.append(tmp2); tmp2=[]
    return tmp

  def looper_empty(x, fill, dtype=int32):
    if isinstance(x, int): return helper.cast([fill] * x, dtype)
    if isinstance(x, list):
      if len(x) == 1: return helper.cast([fill] * x[0], dtype)
      return helper.cast([fill] * len(x), dtype)
    if len(x) <= 2: return helper.cast(helper.getrow(helper, x, fill), dtype) # if onedimentional [2, 4]
    return helper.cast([[helper.getrow(helper, x, fill) for i in range(x[l])] for l in range(len(x) - 2, -1, -1)].pop(), dtype)

  def looper_empty_like(x, fill, dtype=int32):
    if isinstance(x, int): return [fill] * x
    if isinstance(x, list):
      if not isinstance(x[0], list): return [fill] * len(x)
      return [fill for a in range(len(x)) for b in range(len(x[a]))]

  def looper_getrow(x, fill):
    if isinstance(x, int): return [fill] * x
    if isinstance(x, list):
      if len(x) == 1: return [fill] * x[0]
      return [fill] * len(x)

  def looper_max(x, ret=0):
    for i in range(len(x)):
      if isinstance(x[i], (list, tuple)): ret = helper.looper_max(x[i], ret)
      elif ret <= x[i]: ret = x[i]
    return ret

  def looper_min(x, ret=0):
    for i in range(len(x)):
      if isinstance(x[i], (list, tuple)): ret = helper.looper_min(x[i], ret)
      elif ret >= x[i]: ret = x[i]
    return ret

  def looper_maximum(x, y, ret=0):
    tmp=[]; ret=[]
    for i in range(len(x)):
      if builtins.all(isinstance(j, list) for j in [x[i],y[i]]): tmp.append(helper.looper_maximum(x[i], y[i], ret))
      else: tmp.append(y[i]); tmp.append(x[i]); ret.extend(tmp); tmp = []
    return ret

  def looper_minimum(x, y, ret=0):
    tmp=[]; ret=[]
    for i in range(len(x)):
      if builtins.all(isinstance(j, list) for j in [x[i],y[i]]): tmp.append(helper.looper_minimum(x[i], y[i], ret))
      else: tmp.append(y[i]); tmp.append(x[i]); ret.extend(tmp); tmp = []
    return ret

  def looper_isin(x, y, ret):
    for i in range(len(x)):
      if isinstance(y, list) and builtins.all(isinstance(j, list) for j in [x[i],y[i]]): ret = helper.looper_isin(x[i], y[i], ret)
      elif isinstance(x[i], list) and not isinstance(y, list): ret = helper.looper_isin(x[i], y, ret)
      elif isinstance(x[i], tuple): ret = helper.looper_isin(x[i], y, ret)
      else:
        if x[i] == y[i]: ret.append(True)
        else: ret.append(False)
    return ret

  def looper_count(x, ret):
    for i in range(len(x)):
      if isinstance(x[i], (list, tuple)): ret = helper.looper_count(x[i], ret)
      elif (x[i] != 0 and x[i] is not False): ret += 1
    return ret

  def looper_matmul(x, y, dtype=int32):
    if not isinstance(x, (list, tuple)):
      if x and y: return x * y
      elif x and not y: return x
      else: return y
    ret = [helper.looper_matmul(i, y=j) for i, j in zip(x, y)]
    return helper.reshape(ret, -1)

  def looper_assert(x, y):
    ret=[]
    for i in range(len(x)):
      if isinstance(y, list) and builtins.all(isinstance(j, list) for j in [x[i],y[i]]): ret = helper.looper_assert(x[i], y[i])
      elif isinstance(x[i], list) and not isinstance(y, list): ret = helper.looper_assert(x[i], y)
      elif isinstance(x[i], tuple): ret = helper.looper_assert(x[i], y)
      elif isinstance(x[i], ndarray): ret = helper.looper_assert(x[i].tolist(), y[i].tolist())
      else:
        if round(x[i], 8) == round(y[i], 8): ret.append(True)
        else: ret.append(False)
    return ret

  def looper_assert_close(x, y):
    ret=[]
    for i in range(len(x)):
      if isinstance(y, list) and builtins.all(isinstance(j, list) for j in [x[i],y[i]]): ret = helper.looper_assert_close(x[i], y[i])
      elif isinstance(x[i], list) and not isinstance(y, list): ret = helper.looper_assert_close(x[i], y)
      elif isinstance(x[i], tuple): ret = helper.looper_assert_close(x[i], y)
      else:
        if round(x[i], 8) == round(y[i], 8): ret.append(True)
        else: ret.append(False)
        ret.append(abs(abs(x[i]) - abs(y[i])))
        if y[i]: ret.append(abs(abs(x[i]) - abs(y[i])) / abs(y[i]))
    return ret

  def looper_assert_close_cls(x, y):
    ret=[]
    for i in range(len(x)):
      if isinstance(y, list) and builtins.all(isinstance(j, list) for j in [x[i],y[i]]): ret = helper.looper_assert_close_cls(x[i], y[i])
      elif isinstance(x[i], list) and not isinstance(y, list): ret = helper.looper_assert_close_cls(x[i], y)
      elif isinstance(x[i], tuple): ret = helper.looper_assert_close_cls(x[i], y)
      else:
        if round(x[i], 8) == round(y[i], 8): ret.append(True)
        else: ret.append(False)
        ret.append(abs(abs(x[i]) - abs(y[i])))
        if y[i]: ret.append(abs(abs(x[i]) - abs(y[i])) / abs(y[i]))
        ret.append(abs(y[i]))
    return ret

  def reshape(col, shape):
    ncols, nrows, ret, row = 0, 0, [], []
    if shape == -1:
      if not isinstance(col, (list, tuple)): ncols, nrows = col, 1
      else: ncols, nrows = len(col), 1
    elif isinstance(shape, tuple): nrows, ncols = shape
    else: ncols, nrows = len(col), 1
    for r in range(nrows):
      for c in range(ncols):
        if shape == -1 and isinstance(col, list) and not isinstance(col[c], (float, int)): row.extend(helper.reshape(col[c], -1))
        elif shape == -1: row.extend(col); break
        else: row.append(col[ncols * r + c])
      if shape == -1: ret.extend(row)
      else: ret.append(row); row=[]
    return ret

  def read_magic(f): return f.read(len(helper.MAGIC_PREFIX)) == helper.MAGIC_PREFIX
  def read_header_type(f): return int.from_bytes(f.read(1), "big"), int.from_bytes(f.read(1), "big")
  def read_header_pack(t): return helper._header_size_info.get(t)[0]
  def read_header_enc(t): return helper._header_size_info.get(t)[1]
  def read_header_len(f, p): return struct.unpack(helper.read_header_pack(p), f.read(2))[0]
  def read_header(f, t):
    h = f.read(helper.read_header_len(f, t)).decode(helper.read_header_enc(t))
    return ast.literal_eval(h)['shape'], ast.literal_eval(h)['descr']
  def read_body(f, l): return [int.from_bytes(f.read(8), "little") for _ in range(0, l, 8)]

  def write_magic(f): f.write(helper.MAGIC_PREFIX)
  def write_header_type(f, t): f.write(t[0].to_bytes(1, byteorder ='big')); f.write(t[1].to_bytes(1, byteorder ='big'))
  def write_header_len(f, p, header): f.write(struct.pack('<H', len(header)))
  def write_header(f, t, header): f.write(header.encode())
  def write_body(f, x):
    for i in x: f.write(i.to_bytes(8,'little'))
