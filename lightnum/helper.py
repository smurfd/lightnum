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
  _header_size_info = {(1, 0): ('<H', 'latin1'), (2, 0): ('<I', 'latin1'), (3, 0): ('<I', 'utf8')}
  def typ(x, dtype=int32): return dtype(x).value if not isinstance(x, int) else x
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
  def looper_log2(x, dtype=float64): return math.log2(x) if not isinstance(x, list) else [helper.looper_log2(i) for i in x]
  def looper_exp(x, dtype=float64): return math.exp(x) if not isinstance(x, list) else [helper.looper_exp(i) for i in x]
  def looper_exp2(x, dtype=float32): return 2 ** x if not isinstance(x, list) else [helper.looper_exp2(i) for i in x]
  def looper_cbrt(x, dtype=float32): return helper.cbrt(x) if not isinstance(x, list) else [helper.looper_cbrt(i) for i in x]
  def looper_sum(x, dtype=float16): return builtins.sum(x) if not isinstance(x, list) else [helper.looper_sum(i) for i in x]
  def looper_mod(x, y, dtype=float32): return (x%y) if not isinstance(x, (list, tuple)) else [helper.looper_mod(i, y=j) for i, j in zip(x, y)]
  def looper_add(x, y, dtype=float32): return (x+y) if not isinstance(x, (list, tuple)) else [helper.looper_add(i, y=j) for i, j in zip(x, y)]
  def looper_subtract(x, y, dtype=float32): return x - y if not isinstance(x, (list, tuple)) else [helper.looper_subtract(i, y=j) for i, j in zip(x, y)]
  def looper_prod(x, dtype=int32): return math.prod(x) if not isinstance(x, list) else [helper.looper_prod(i) for i in x]
  def looper_cos(x, dtype=float64): return math.cos(x) if not isinstance(x, list) else [helper.looper_cos(i) for i in x]
  def looper_ceil(x, dtype=float32): return math.ceil(x) if not isinstance(x, list) else [helper.looper_ceil(i) for i in x]
  def looper_copy(x, dtype=float32): return cp.copy(x) if not isinstance(x, list) else [helper.looper_copy(i) for i in x]
  def looper_where(condition, x, y, dtype=float32): return [xv if c else yv for c, xv, yv in zip(condition, x, y)] if not isinstance(x, list) else [helper.looper_where(condition, i, j) for i,j in zip(x, y)]
  def looper_sqrt(x, dtype=float32): return math.sqrt(x) if not isinstance(x, list) else [helper.looper_sqrt(i) for i in x]
  def looper_arctan2(x, y, dtype=float32): return math.atan2(x, y) if not isinstance(x, (list, tuple)) else [helper.looper_arctan2(i, y=j) for i, j in zip(x, y)]
  def looper_transpose(x, axes=None): return [[row[i] for row in x] for i in range(len(x[0]))] #TODO: axes
  def looper_stack(x, axis=0): return [i for i in x] #TODO: axis
  def looper_expand_dims(x, axis, axisco=0): return [x] if axisco == axis else [helper.looper_expand_dims(x[i], axis, axisco + 1) for i in range(len(x))]
  def looper_flip(x, dtype=int32): return [helper.looper_flip(i) for i in x] if isinstance(x[0], list) else x[::-1]
  def looper_squeeze(x, axis=0): return x if not isinstance(x[0][0], list) else [helper.looper_squeeze(y, axis) for y in x].pop() #TODO: axis
  def looper_clip(x, x_min, x_max): return (x_min if x < x_min else x_max) if not isinstance(x, list) else [helper.looper_clip(y, x_min, x_max) for y in x]
  def looper_broadcast_to(x, y): return x if len(y) == 1 else [x for i in range(y[0])]
  def looper_frombuffer(buf, dtype=int32): return [helper.typ(i, dtype=dtype) for i in buf]
  def looper_sin(x, dtype=float64): return math.sin(x) if not isinstance(x, list) else [helper.looper_sin(i) for i in x]
  def looper_reciprocal(x, dtype=float64): return 1/(x) if not isinstance(x, list) else [helper.looper_reciprocal(i) for i in x]
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

  def looper_median(x, dtype=int32, ret=0):
    for i in range(len(x)):
      if isinstance(x[i], (list, tuple)): ret = helper.looper_median(x[i], ret)
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

  # BARF
  def looper_pad(x, y, mode='constant', **kwargs):
    ret = []
    app = [x[i] for i in range(len(x))]
    if mode == 'constant':
      ret.extend(kwargs['constant_values'][0] for i in range(y[0] if isinstance(y, tuple) else y)); ret.extend(app); ret.extend(kwargs['constant_values'][1] for i in range(y[1] if isinstance(y, tuple) else y))
    elif mode == 'edge':
      ret.extend(x[0] for i in range(y[0] if isinstance(y, tuple) else y)); ret.extend(app); ret.extend(x[len(x)-1] for i in range(y[1] if isinstance(y, tuple) else y))
    elif mode == 'linear_ramp':
      l1 = y[0] if isinstance(y, tuple) else y
      l2 = y[1] if isinstance(y, tuple) else y
      step=int(kwargs['end_values'][0] / l1)
      start=x[0] if x[0] > kwargs['end_values'][0] else kwargs['end_values'][0]
      end=kwargs['end_values'][0] if kwargs['end_values'][0] < x[0] else x[0]
      if start > end: step = -1*step
      ret.extend(i for i in range(start, end, step)); ret.extend(app)
      step=int((x[len(x)-1]-kwargs['end_values'][1]) / l2)
      start=x[len(x)-1] if x[len(x)-1] > kwargs['end_values'][1] else kwargs['end_values'][1]
      end=kwargs['end_values'][1]
      start = (start-step)
      if start > end: step = -1 * step
      md = (list(range(start, end, step))[len(list(range(start, end, step)))-1]-end) if list(range(start, end, step))[len(list(range(start, end, step)))-1] > end else (end-list(range(start, end, step))[len(list(range(start, end, step)))-1])
      ret.extend(i for i in range(start - md, end - md, step))
    elif mode == 'maximum':
      maxv = x[0]
      for i in range(len(x)):
        if maxv < x[i]: maxv = x[i]
      ret.extend(maxv for i in range(y[0] if isinstance(y, tuple) else y)); ret.extend(app); ret.extend(maxv for i in range(y[1] if isinstance(y, tuple) else y))
    elif mode == 'mean':
      meanv = 0
      for i in range(len(x)): meanv += x[i]
      meanv = int(meanv / len(x))
      ret.extend(meanv for i in range(y[0] if isinstance(y, tuple) else y)); ret.extend(app); ret.extend(meanv for i in range(y[1] if isinstance(y, tuple) else y))
    elif mode == 'median':
      if isinstance(len(x)/2, int): meadv = x[len(x)/2]
      else: meadv = int(math.ceil((x[int(len(x)/2)] + x[int(len(x)/2)-1]) / 2))
      ret.extend(meadv for i in range(y[0] if isinstance(y, tuple) else y))
      ret.extend(x[i] for i in range(len(x)))
      ret.extend(meadv for i in range(y[1] if isinstance(y, tuple) else y))
    elif mode == 'minimum':
      minv = x[0]
      for i in range(len(x)):
        if minv > x[i]: minv = x[i]
      ret.extend(minv for i in range(y[0] if isinstance(y, tuple) else y)); ret.extend(app); ret.extend(minv for i in range(y[1] if isinstance(y, tuple) else y))
    elif mode == 'reflect': # TODO reflect_type
      ret.extend(x[(y[0] if isinstance(y, tuple) else y)-i] for i in range(y[0] if isinstance(y, tuple) else y)); ret.extend(app); ret.extend(x[(y[1] if isinstance(y, tuple) else y)-i+1] for i in range(y[1] if isinstance(y, tuple) else y))
    elif mode == 'symmetric': # TODO reflect_type
      ret.extend(x[(y[0] if isinstance(y, tuple) else y)-i-1] for i in range(y[0] if isinstance(y, tuple) else y)); ret.extend(app); ret.extend(x[len(x)-(y[1] if isinstance(y, tuple) else y)-i+1] for i in range(y[1] if isinstance(y, tuple) else y))
    elif mode == 'wrap':
      ret.extend(x[len(x)-i] for i in range(y[1] if isinstance(y, tuple) else y, 0, -1)); ret.extend(app); ret.extend(x[i] for i in range(y[0] if isinstance(y, tuple) else y))
    elif mode == 'empty':
      ret.extend(None for i in range(y[0] if isinstance(y, tuple) else y)); ret.extend(app); ret.extend(None for i in range(y[1] if isinstance(y, tuple) else y))
    return ret

  def reshape(col, shape):
    ncols, nrows, ret, row = 0, 0, [], []
    if shape == -1:
      if not isinstance(col, (list, tuple)): ncols, nrows = col, 1
      else: ncols, nrows = len(col), 1
      for r in range(nrows):
        for c in range(ncols):
          if isinstance(col, list) and not isinstance(col[c], (float, int)): row.extend(helper.reshape(col[c], -1))
          else: row.extend(col); break
        ret.extend(row)
        row=[]
      return ret
    if isinstance(shape, tuple): nrows, ncols = shape
    else: ncols, nrows = len(col), 1
    for r in range(nrows):
      for c in range(ncols):
        row.append(col[ncols * r + c])
      ret.append(row)
      row=[]
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
  def write_body(f, x): [f.write(i.to_bytes(8,'little')) for i in x]
