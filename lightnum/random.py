#!/usr/bin/env python3
from __future__ import annotations
from typing import List, Any, Callable, Type, Optional, Union
from lightnum.array import array, ndarray, asarray
from lightnum.helper import helper
from lightnum.dtypes import int32, float32, dtype
import random as rnd

h = helper()
class random():
  class default_rng():
    x: int = 0
    y: int = 0
    def __init__(self, x: int, y: int) -> None:
      self.x = x
      self.y = y
    def __call__(self) -> Any: return rnd.randint(self.x, self.y)
    def random(self, size: int, dtype: dtype=float32) -> Any: return random.randn(random(), size)
  def seed(self, x: int, dtype: dtype=int32) -> Any: return rnd.seed(x)
  class randint():
    x: int = 0
    y: int = 0
    ret: List[Any]
    def __init__(self, x: int, y: Optional[int]=None, size: Optional[int]=None, dtype: dtype=int32):
      if size and y: self.ret = [rnd.randint(x, y) for i in range(size)]
      if size: self.ret = [rnd.randint(0, x) for i in range(size)]
      if y: self.ret = [rnd.randint(int(x), int(y))]
    def __call__(self) -> Any: return rnd.randint(self.x, self.y)
    def tolist(self) -> List[Any]: return list(self.ret)
  def randn(self, *args: Any, dtype: dtype=int32) -> ndarray:
    ret: List[Any] = []
    if isinstance(args, tuple) and len(args) == 1:
      if len(args[0]) != 1:
        for j in range(len(args[0])):
          for i in range(args[0][j]):
            if i != j: ret.append(i)
      else:
        for i in range(len(args[0])): ret.append(i)
    else:
      for i in args: ret.append(i)
    return ndarray(h.looper_empty(ret, fill=rnd.random(), dtype=float32))
  def random(self, size:int, dtype: dtype=float32) -> Any: return random.randn(self, size)
  def rand(self, *args: Any, dtype: dtype=int32) -> Any: return random.randn(self, args, dtype=dtype)
  def RandomState(self, x:int, dtype: dtype=int32) -> Any: return random.randn(self, x)
  class uniform():
    ret: Union[List[Any], int] = []
    def __init__(self, x: int, y: int, size: int) -> None: self.ret = [rnd.randint(x, y) for _ in range(size)] if size else rnd.randint(x, y)
    def tolist(self) -> Any: return self.ret
  class choice():
    ret: List[Any] | int = []
    def __init__(self, x: Any, size: int=0, dtype: dtype=int32) -> None: self.ret = [rnd.choice(x) for _ in range(size)] if size else rnd.choice(x)
    def tolist(self) -> Any: return self.ret
