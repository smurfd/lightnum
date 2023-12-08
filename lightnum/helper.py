from lightnum.dtypes import int32, float32
import copy as cp
import builtins
import math

class helper():
  def typ(x, dtype=int32): return dtype(x).value
  def sum(x, y): return x + y
  def mod(x, y): return x % y
  def exp2(x): return 2 ** x
  def cbrt(x): return round(x**(1 / 3.), 2)
  def getrow(self, x, fill=0): return [helper.looper_getrow(x[-1], fill=fill) for _ in range(x[len(x) - 2])]

  # helper functions to loop through multidimentional lists/tuples
  def looper_log(x, dtype=int32):
    if not isinstance(x, list): return math.log(x)
    return [helper.looper_log(i) for i in x]

  def looper_exp(x, dtype=int32):
    if not isinstance(x, list): return math.exp(x)
    return [helper.looper_exp(i) for i in x]

  def looper_exp2(x, dtype=int32):
    if not isinstance(x, list): return helper.exp2(x)
    return [helper.looper_exp2(i) for i in x]

  def looper_cbrt(x, dtype=int32):
    if not isinstance(x, list): return helper.cbrt(x)
    return [helper.looper_cbrt(i) for i in x]

  def looper_sum(x, dtype=int32):
    if not isinstance(x, list): return builtins.sum(x)
    return [helper.looper_sum(i) for i in x]

  def looper_mod(x, y, dtype=int32):
    if not isinstance(x, (list, tuple)): return helper.mod(x, y)
    return [helper.looper_mod(i, y=j) for i, j in zip(x, y)]

  def looper_prod(x, dtype=int32):
    if not isinstance(x, list): return math.prod(x)
    return [helper.looper_prod(i) for i in x]

  def looper_cos(x, dtype=int32):
    if not isinstance(x, list): return math.cos(x)
    return [helper.looper_cos(i) for i in x]

  def looper_sqrt(x, dtype=int32):
    if not isinstance(x, list): return math.sqrt(x)
    return [helper.looper_sqrt(i) for i in x]

  def looper_arctan2(x, y, dtype=int32):
    if not isinstance(x, (list, tuple)): return math.atan2(x, y)
    return [helper.looper_arctan2(i, y=j) for i, j in zip(x, y)]

  def looper_ceil(x, dtype=int32):
    if not isinstance(x, list): return math.ceil(x)
    return [helper.looper_ceil(i) for i in x]

  def looper_copy(x, dtype=int32):
    if not isinstance(x, list): return cp.copy(x)
    return [helper.looper_copy(i) for i in x]

  def looper_where(condition, x, y, dtype=int32):
    if not isinstance(x, list): return [xv if c else yv for c, xv, yv in zip(condition, x.tolist(), y)]
    return [helper.looper_where(condition, i, j) for i,j in zip(x, y)]

  def looper_nonzero(x):#, x, y=[], dtype=int32):
    cc = [not x1 for x1 in x[0]]
    cc.extend([not x1 for x1 in x[1]])
    print(x[0])
    print(x[1])
    print(cc)
    #return helper.looper_where(c, x, x)
    if not isinstance(x, list): return [xv if c else yv for c, xv, yv in zip(cc, x, x)]
    return [helper.looper_where(cc, i, j) for i,j in zip(x, x)]

  def looper_arange(start, stop=0, step=1, dtype=int32):
    from lightnum.array import array, ndarray
    if stop: return ndarray([i for i in range(start, stop, step)])
    return ndarray([i for i in range(0, start, step)])

  def looper_argmax(x, axis=None):
    if not axis:
      y = helper.reshape(x, -1)
      return y.index(max(y))
    if not isinstance(x[0], list): return x.index(max(x))
    return [helper.looper_argmax(y, axis) for y in x]

  def looper_transpose(x, axes=None): return [[row[i] for row in x] for i in range(len(x[0]))] #TODO: axes
  def looper_stack(x, axis=0): return [i for i in x] #TODO: axis
  def looper_vstack(x): return [i for i in x]
  def looper_squeeze(x, axis=0): #TODO: axis
    if not isinstance(x[0][0], list): return x
    return [helper.looper_squeeze(y, axis) for y in x].pop()

  def looper_clip(x, x_min, x_max):
    if not isinstance(x, list):
      if x < x_min: x = x_min
      if x > x_max: x = x_max
      return x
    return [helper.looper_clip(y, x_min, x_max) for y in x]

  def looper_unique(x): return list(set(sorted(helper.reshape(x, -1))))

  def looper_broadcast_to(x, y):
    if len(y) == 1: return x
    return [x for i in range(y[0])]

  def looper_cumsum(x, s=0,dtype=int32):
    a = helper.reshape(x, -1)
    return [builtins.sum(a[0:i:1]) for i in range(0, len(a)+1)][1:]

  def looper_add(x, dtype=int32, ret=0):
    for i in range(len(x)):
      if isinstance(x[i], (list, tuple)): ret = helper.looper_add(x[i], ret)
      else: ret += x[i]
    return ret

  def looper_expand_dims(x, axis, axisco=0):
    if axisco == axis: return [x]
    return [helper.looper_expand_dims(x[i], axis, axisco + 1) for i in range(len(x))]

  def looper_flip(x, dtype=int32):
    if isinstance(x[0], list): return [helper.looper_flip(i) for i in x]
    return x[::-1]

  def looper_split(x, y, dtype=int32):
    from lightnum.array import array, ndarray
    if not isinstance(y, list) and not isinstance(x, list): return [ndarray(x.tolist()[i:i+len(x.tolist())//y]) for i in range(0, len(x.tolist()), len(x.tolist())//y)]
    elif not isinstance(y, list): return [ndarray(x[i:i+(len(x)//y)]) for i in range(0, len(x), len(x)//y)]
    return [helper.looper_split(i, y) for i in x]

  def looper_tile(x, y, dtype=int32):
    from lightnum.array import array, ndarray
    tmp, tmp2=[], []
    if not isinstance(y, tuple):
      for _ in range(y): tmp.extend(x)
      return ndarray(tmp)
    a,b = y
    for a1 in range(a):
      for b1 in range(b): tmp2.extend(x)
      tmp.append(tmp2); tmp2=[]
    return tmp

  def looper_concatenate(x, dtype=int32): return [i for j in range(len(x)) for i in x[j]]

  def looper_empty(x, fill):
    if isinstance(x, int): return [fill] * x
    if isinstance(x, list):
      if len(x) == 1: return [fill] * x[0]
      return [fill] * len(x)
    if len(x) <= 2: return helper.getrow(helper, x, fill) # if onedimentional [2, 4]
    return [[helper.getrow(helper, x, fill) for i in range(x[l])] for l in range(len(x) - 2, -1, -1)].pop()

  def looper_empty_like(x, fill):
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

  def looper_outer(x, y): return [[x1*y1 for y1 in y] for x1 in x]

  def looper_matmul(x, y, dtype=int32):
    from lightnum.lightnum import reshape
    if not isinstance(x, (list, tuple)):
      if x and y: return x * y
      elif x and not y: return x
      else: return y
    ret = [helper.looper_matmul(i, y=j) for i, j in zip(x, y)]
    return reshape(ret, -1)

  def looper_assert(x, y):
    from lightnum.array import array, ndarray
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
