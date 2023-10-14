from lightnum.dtypes import int32,float32
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

  # tuple (1,2,3,4,5)
  #                ^- # per row
  #              ^- # rows per box
  #            ^- # boxes
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

  def looper_mul(x, dtype=int32):
    if not isinstance(x, list): return math.prod(x)
    return [helper.looper_mul(i) for i in x]

  def looper_cos(x, dtype=int32):
    if not isinstance(x, list): return math.cos(x)
    return [helper.looper_cos(i) for i in x]

  def looper_sqrt(x, dtype=int32):
    if not isinstance(x, list): return math.sqrt(x)
    return [helper.looper_sqrt(i) for i in x]

  def looper_atan2(x, y, dtype=int32):
    if not isinstance(x, (list, tuple)): return math.atan2(x, y)
    return [helper.looper_atan2(i, y=j) for i, j in zip(x, y)]

  def looper_ceil(x, dtype=int32):
    if not isinstance(x, list): return math.ceil(x)
    return [helper.looper_ceil(i) for i in x]

  def looper_cp(x, dtype=int32):
    if not isinstance(x, list): return cp.copy(x)
    return [helper.looper_cp(i) for i in x]

  def looper_add(x, dtype=int32, ret=0):
    for i in range(len(x)):
      if isinstance(x[i], (list, tuple)): ret = helper.looper_add(x[i], ret)
      else: ret += x[i]
    return ret

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

  def looper_maxi(x, y, ret=0):
    tmp=[]; ret=[]
    for i in range(len(x)):
      if builtins.all(isinstance(j, list) for j in [x[i],y[i]]): tmp.append(helper.looper_maxi(x[i], y[i], ret))
      else: tmp.append(y[i]); tmp.append(x[i]); ret.extend(tmp); tmp = []
    return ret

  def looper_mini(x, y, ret=0):
    tmp=[]; ret=[]
    for i in range(len(x)):
      if builtins.all(isinstance(j, list) for j in [x[i],y[i]]): tmp.append(helper.looper_mini(x[i], y[i], ret))
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

  def looper_assert(x, y):
    ret=[]
    for i in range(len(x)):
      if isinstance(y, list) and builtins.all(isinstance(j, list) for j in [x[i],y[i]]): ret = helper.looper_assert(x[i], y[i])
      elif isinstance(x[i], list) and not isinstance(y, list): ret = helper.looper_assert(x[i], y)
      elif isinstance(x[i], tuple): ret = helper.looper_assert(x[i], y)
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
