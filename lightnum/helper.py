#!/usr/bin/env python3
from __future__ import annotations
from typing import List, Any, BinaryIO, Tuple, Union, Dict, Iterable, Sized
from lightnum.dtypes import int32, float16, float32, float64, types, dtype
from lightnum.array import ndarray
import itertools, builtins, ctypes, struct, math

class helper:
  MAGIC_PREFIX = b'\x93NUMPY'
  _header_size_info: Dict[Tuple[int, int], Tuple[str, str]] = {(1, 0): ('<H', 'latin1'), (2, 0): ('<I', 'latin1'), (3, 0): ('<I', 'utf8')}
  def __init__(self) -> None: pass
  def cbrt(self, x: int) -> Any: return round(x**(1 / 3.), 2)
  def format_float(self, x: float) -> Any: return float(('%i' if x == int(x) else '%s') % x)
  def cast(self, x: Union[float, int, List[Any]], dtype:dtype = float64) -> Any: return self.looper_cast(x, dtype=dtype) if str(dtype) != types[dtype(x)] else x
  def zero_row_len(self, x: List[Any], l: int) -> List[Any]:
    for i in range(l):
      if i >= 0 and i < len(x): x[i] = 0
    return x
  def if_round_abs(self, x: float, y: float, cls: bool=False) -> List[Any]:
    ret: List[Any] = []
    if round(x, 8) == round(y, 8): ret.append(True)
    else: ret.append(False)
    ret.append(abs(abs(x) - abs(y)))
    if y: ret.append(abs(abs(x) - abs(y)) / abs(y))
    if cls: ret.append(abs(y))
    return ret

  # helper functions to loop through multidimentional lists/tuples
  def looper_cast(self, x: Union[float, int, List[Any]], dtype: dtype = float64) -> Any:
    if isinstance(x, list): return [self.looper_cast(i, dtype) for i in x]
    a = ctypes.cast(ctypes.pointer(dtype(x)(round(x, 8))), ctypes.POINTER(dtype(x))).contents.value
    b = (not isinstance(x, list) and x != float(int(x)))
    return (round(a, 8 if len(str(x)) > 8 else len(str(x))) if b else int(a) if dtype in [float16, float32, float64] else a)
  def looper_transpose(self, x: List[Any], axes: Union[Any, int]=None) -> List[Any]: return [[row[i] for row in x] for i in range(len(x[0]))] #TODO: axes
  def looper_stack(self, x: List[Any], axis: int = 0) -> List[Any]: return [i for i in x] #TODO: axis
  def looper_expand_dims(self, x: List[Any], axis: int, axisco: int = 0) -> List[Any]: return [x] if axisco == axis else [self.looper_expand_dims(x[i], axis, axisco + 1) for i in range(len(x))]
  def looper_flip(self, x: List[Any], dtype: dtype = int32) -> List[Any]: return [self.looper_flip(i) for i in x] if isinstance(x[0], list) else x[::-1]
  def looper_squeeze(self, x: List[Any], axis: int = 0) -> List[Any]: return x if not isinstance(x[0][0], list) else [self.looper_squeeze(y, axis) for y in x].pop() #TODO: axis
  def looper_clip(self, x: Union[List[Any], int], x_min: int, x_max: int) -> Union[Any, Any]: return (x_min if x < x_min else x_max) if not isinstance(x, list) else [self.looper_clip(y, x_min, x_max) for y in x]
  def looper_broadcast_to(self, x: List[Any], y: Union[List[Any], Tuple[int, int]]) -> List[Any]: return x if len(y) == 1 else [x for i in range(y[0])]
  def looper_nonzero(self, x: List[Any]) -> Tuple[ndarray, ...]:
    ret1: List[Any] = []
    ret2: List[Any] = []
    if isinstance(x[0], list): [[(ret1.append(i), ret2.append(ii)) for ii in range(len(x[i])) if x[i][ii]] for i in range(len(x)) if isinstance(x[i], list)] #type: ignore
    return tuple([ndarray(ret1)] + [ndarray(ret2)])

  def looper_arange(self, start: int, stop: int = 0, step: int = 1, dtype: dtype = int32) -> Union[List[Any], ndarray]:
    if stop: return ndarray([i for i in range(start, stop, step)])
    return ndarray([i for i in range(0, start, step)])

  def looper_argmax(self, x: List[Any], axis: int = 0) -> Union[int, List[Any]]:
    if not axis:
      y = self.reshape(x, -1)
      return y.index(max(y))
    return x.index(max(x)) if not isinstance(x[0], list) else [self.looper_argmax(y, axis) for y in x]

  def looper_cumsum(self, x: List[Any], s: int = 0, dtype: dtype = int32) -> List[Any]:
    a = self.reshape(x, -1)
    return [builtins.sum(a[0:i:1]) for i in range(0, len(a)+1)][1:]

  def looper_median(self, x: List[Any], dtype: dtype = int32) -> int:
    ret = 0
    for i in range(len(x)):
      if isinstance(x[i], (list, tuple)): ret = self.looper_median(x[i])
      else: ret += x[i]
    return ret

  def looper_split(self, x: List[Any], y: List[Any], dtype: dtype = int32) -> List[Any]:
    if not isinstance(y, list) and not isinstance(x, list): return [ndarray(x.tolist()[i:i+len(x.tolist())//y]) for i in range(0, len(x.tolist()), len(x.tolist())//y)]
    elif not isinstance(y, list): return [ndarray(x[i:i+(len(x)//y)]) for i in range(0, len(x), len(x)//y)]
    return [self.looper_split(i, y) for i in x]

  def looper_tile(self, x: List[Any], y: Union[List[Any], Tuple[Any, Any]], dtype: dtype = int32) -> List[Any]:
    tmp: List[Any] = []
    tmp2: List[Any] = []
    if not isinstance(y, tuple):
      for _ in range(y if not isinstance(y, (list, tuple)) else len(y)): tmp.extend(x)
      return ndarray(tmp) #type: ignore
    a,b = y
    for _ in range(a):
      for _ in range(b): tmp2.extend(x)
      tmp.append(tmp2)
      tmp2=[]
    return tmp

  def looper_empty(self, x: List[Any], fill: Union[float, int], dtype: dtype = int32) -> List[Any]:
    if isinstance(x, (int, float)): return [fill]*x
    elif len(x)==1: return([fill]*x[0])
    elif isinstance(x, (list, tuple)) and isinstance(x[0], (list, tuple)): return [self.looper_empty(x[i], fill=fill, dtype=dtype) for i in range(len(x))]
    r: List[Any] = []
    ret: List[Any] = []
    ret1: List[Any] = []
    for j, idx in enumerate(itertools.product(*[range(s) for s in x[1::]])): #start from 1 since we will fill a row of x[0] len for each
      if not j%(x[0]) and j:
        ret.append(ret1)
        ret1=[]
      if not j%(x[0]*x[0]) and j:
        r.append(ret)
        ret=[]
      ret1.append([fill]*x[0])
    if r and x[0] > 2:
      ret.append(ret1)
      r.append(ret)
    if not r and x[0] > 2:
      for _ in range(x[0] - len(ret)): r.extend(ret)
      r.append(ret1)
    if not r and not ret: r.extend(ret1)
    return r

  def looper_empty_like(self, x: List[Any], fill: Union[float, int], dtype: dtype = int32) -> List[Any]:
    if isinstance(x, int): return [fill] * x
    elif isinstance(x, list):
      if not isinstance(x[0], list): return [fill] * len(x)
      return [fill for a in range(len(x)) for b in range(len(x[a]))]

  def looper_max(self, x: List[Any], ret: Union[float, int, List[Any]] = 0) -> Union[float, int, List[Any]]:
    for i in range(len(x)):
      if isinstance(x[i], (list, tuple)): ret = self.looper_max(x[i], ret)
      elif ret <= x[i]: ret = x[i]
    return ret

  def looper_min(self, x: List[Any], ret: Union[float, int, List[Any]] = 0) -> Union[float, int, List[Any]]:
    for i in range(len(x)):
      if isinstance(x[i], (list, tuple)): ret = self.looper_min(x[i], ret)
      elif ret >= x[i]: ret = x[i]
    return ret

  def looper_maximum(self, x: List[Any], y: List[Any]) -> List[Any]:
    tmp, ret = [], []
    for i in range(len(x)):
      if builtins.all(isinstance(j, list) for j in [x[i],y[i]]): tmp.append(self.looper_maximum(x[i], y[i]))
      else: tmp.extend([x[i]]) if x[i] >= y[i] else tmp.extend([y[i]])
    ret.extend(tmp)
    return ret

  def looper_minimum(self, x: List[Any], y: List[Any]) -> List[Any]:
    tmp, ret = [], []
    for i in range(len(x)):
      if builtins.all(isinstance(j, list) for j in [x[i],y[i]]): tmp.append(self.looper_minimum(x[i], y[i]))
      else: tmp.extend([x[i]]) if x[i] <= y[i] else tmp.extend([y[i]])
    ret.extend(tmp)
    return ret

  def looper_isin(self, x: List[Any], y: List[Any]) -> List[Any]:
    ret = []
    for i in range(len(x)):
      if isinstance(y, list) and builtins.all(isinstance(j, list) for j in [x[i],y[i]]): ret = self.looper_isin(x[i], y[i])
      elif isinstance(x[i], list) and not isinstance(y, list): ret = self.looper_isin(x[i], y)
      elif isinstance(x[i], tuple): ret = self.looper_isin(x[i], y)
      else: ret.append(True) if x[i] == y[i] else ret.append(False)
    return ret

  def looper_count(self, x: Union[int, Iterable[object], List[Any], Sized], ret: int = 0) -> Union[int, Iterable[object]]:
    for i in range(len(x)): #type: ignore
      if isinstance(x[i], (list, tuple)): ret = self.looper_count(x[i], ret) #type: ignore
      elif (x[i] != 0 and x[i] is not False): ret += 1 #type: ignore
    return ret

  # BARF
  def looper_pad(self, x: List[Any], y: Union[List[Any], Tuple[Any, Any]], mode: str ='constant', **kwargs: Any) -> List[Any]:
    ret: List[Any] = []
    app = [x[i] for i in range(len(x))]
    if mode == 'constant':
      ret.extend(kwargs['constant_values'][0] for i in range(y[0] if isinstance(y, tuple) else y))
      ret.extend(app)
      ret.extend(kwargs['constant_values'][1] for i in range(y[1] if isinstance(y, tuple) else y))
    elif mode == 'edge':
      ret.extend(x[0] for i in range(y[0] if isinstance(y, tuple) else y))
      ret.extend(app)
      ret.extend(x[len(x)-1] for i in range(y[1] if isinstance(y, tuple) else y))
    elif mode == 'linear_ramp':
      l1 = y[0] if isinstance(y, tuple) else y
      l2 = y[1] if isinstance(y, tuple) else y
      step=int(kwargs['end_values'][0] / l1)
      start=x[0] if x[0] > kwargs['end_values'][0] else kwargs['end_values'][0]
      end=kwargs['end_values'][0] if kwargs['end_values'][0] < x[0] else x[0]
      if start > end: step = -1*step
      ret.extend(i for i in range(start, end, step))
      ret.extend(app)
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
      ret.extend(maxv for i in range(y[0] if isinstance(y, tuple) else y))
      ret.extend(app)
      ret.extend(maxv for i in range(y[1] if isinstance(y, tuple) else y))
    elif mode == 'mean':
      meanv: Union[int, float] = 0
      for i in range(len(x)): meanv += x[i]
      meanv = int(meanv / len(x))
      ret.extend(meanv for i in range(y[0] if isinstance(y, tuple) else y))
      ret.extend(app)
      ret.extend(meanv for i in range(y[1] if isinstance(y, tuple) else y))
    elif mode == 'median':
      if isinstance(len(x)/2, int): meadv = x[len(x)/2] #type: ignore
      else: meadv = int(math.ceil((x[int(len(x)/2)] + x[int(len(x)/2)-1]) / 2))
      ret.extend(meadv for i in range(y[0] if isinstance(y, tuple) else y))
      ret.extend(x[i] for i in range(len(x)))
      ret.extend(meadv for i in range(y[1] if isinstance(y, tuple) else y))
    elif mode == 'minimum':
      minv = x[0]
      for i in range(len(x)):
        if minv > x[i]: minv = x[i]
      ret.extend(minv for i in range(y[0] if isinstance(y, tuple) else y))
      ret.extend(app)
      ret.extend(minv for i in range(y[1] if isinstance(y, tuple) else y))
    elif mode == 'reflect': # TODO reflect_type
      ret.extend(x[(y[0] if isinstance(y, tuple) else y)-i] for i in range(y[0] if isinstance(y, tuple) else y))
      ret.extend(app)
      ret.extend(x[(y[1] if isinstance(y, tuple) else y)-i+1] for i in range(y[1] if isinstance(y, tuple) else y))
    elif mode == 'symmetric': # TODO reflect_type
      ret.extend(x[(y[0] if isinstance(y, tuple) else y)-i-1] for i in range(y[0] if isinstance(y, tuple) else y))
      ret.extend(app)
      ret.extend(x[len(x)-(y[1] if isinstance(y, tuple) else y)-i+1] for i in range(y[1] if isinstance(y, tuple) else y))
    elif mode == 'wrap':
      ret.extend(x[len(x)-i] for i in range(y[1] if isinstance(y, tuple) else y, 0, -1))
      ret.extend(app)
      ret.extend(x[i] for i in range(y[0] if isinstance(y, tuple) else y))
    elif mode == 'empty':
      ret.extend(None for i in range(y[0] if isinstance(y, tuple) else y))
      ret.extend(app)
      ret.extend(None for i in range(y[1] if isinstance(y, tuple) else y))
    return ret

  def reshape(self, col: Union[List[Any], ndarray, int], shape: Union[Tuple[int, int], int, List[int]]) -> List[Any]:
    ncols: int = 0
    nrows: int = 0
    ret: List[Any] = []
    row: List[Any] = []
    if isinstance(shape, tuple): nrows, ncols = shape
    elif not isinstance(col, (list, tuple)): ncols, nrows = col, 1 #type: ignore
    else: ncols, nrows = len(col), 1
    if shape == -1:
      for r in range(nrows):
        for c in range(ncols):
          if isinstance(col, list) and not isinstance(col[c], (float, int)): row.extend(self.reshape(col[c], -1))
          else:
            row.extend(col) #type: ignore
            break
        ret.extend(row)
        row=[]
      return ret
    for r in range(nrows):
      for c in range(ncols): row.append(col[ncols * r + c]) #type: ignore
      ret.append(row)
      row=[]
    return ret

  def read_magic(self, f: BinaryIO) -> Any: return f.read(len(self.MAGIC_PREFIX)) == self.MAGIC_PREFIX
  def read_header_type(self, f: BinaryIO) -> Tuple[int, int]: return int.from_bytes(f.read(1), "big"), int.from_bytes(f.read(1), "big")
  def read_header_pack(self, t: Tuple[int, int]) -> str: return self._header_size_info.get(t)[0] #type: ignore
  def read_header_enc(self, t: Tuple[int, int]) -> str: return self._header_size_info.get(t)[1] #type: ignore
  def read_header_len(self, f: BinaryIO, p: Tuple[int, int]) -> Any: return struct.unpack(self.read_header_pack(p), f.read(2))[0]
  def read_header(self, f: BinaryIO, t: Tuple[int, int]) -> Tuple[Any, Any]:
    import ast
    h = f.read(self.read_header_len(f, t)).decode(self.read_header_enc(t))
    return (ast.literal_eval(h)['shape'], ast.literal_eval(h)['descr'])
  def read_body(self, f: BinaryIO, l: int) -> List[int]: return [int.from_bytes(f.read(8), "little") for _ in range(0, l, 8)]

  def write_magic(self, f: BinaryIO) -> None: f.write(self.MAGIC_PREFIX)
  def write_header_type(self, f: BinaryIO, t: Tuple[int, int]) -> None:
    f.write(t[0].to_bytes(1, byteorder ='big'))
    f.write(t[1].to_bytes(1, byteorder ='big'))
  def write_header_len(self, f: BinaryIO, p: Tuple[int, int], header: str) -> None: f.write(struct.pack('<H', len(header)))
  def write_header(self, f: BinaryIO, t: Tuple[int, int], header: str) -> None: f.write(header.encode())
  def write_body(self, f: BinaryIO, x: List[Any]) -> None: [f.write(i.to_bytes(8,'little')) for i in x]
