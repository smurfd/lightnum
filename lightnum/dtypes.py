#!/usr/bin/env python3
from __future__ import annotations
from typing import List, Optional, Any, Type
import ctypes, math

cts = [ctypes.c_float, ctypes.c_float, ctypes.c_double, ctypes.c_int8, ctypes.c_int16, ctypes.c_int32, ctypes.c_int64, ctypes.c_uint8, ctypes.c_uint16, ctypes.c_uint32, ctypes.c_uint64, ctypes.c_bool]
class ctstruct:
  _fields_: List[Any] = []
  def __init__(self, dtype: Type[Any]=ctypes.c_uint8) -> None:
    self._fields_ = list([dtype, dtype])
    self.type = self._fields_[1]
  def __eq__(self, other: ctstruct) -> bool: # type: ignore[override] # ==
    for fld in self._fields_:
      if not isinstance(fld, tuple(cts)):
        if str(self._fields_[1]) == str(other._fields_[1]): return True
      else:
        if getattr(self, fld[0]) == getattr(other, fld[0]): return True
    return False

  def __ne__(self, other: ctstruct) -> bool: # type: ignore[override] # !=
    for fld in self._fields_:
      if not isinstance(fld, tuple(cts)):
        if str(self._fields_[1]) != str(other._fields_[1]): return True
      else:
        if getattr(self, fld[0]) != getattr(other, fld[0]): return True
    return False

  def __lt__(self, other: ctstruct) -> bool: # <
    for fld in self._fields_:
      if not isinstance(fld, tuple(cts)):
        if str(self._fields_[1]) < str(other._fields_[1]): return True
      else:
        if getattr(self, fld[0]) < getattr(other, fld[0]): return True
    return False

  def __le__(self, other: ctstruct) -> bool: # <=
    for fld in self._fields_:
      if not isinstance(fld, tuple(cts)):
        if str(self._fields_[1]) <= str(other._fields_[1]): return True
      else:
        if getattr(self, fld[0]) <= getattr(other, fld[0]): return True
    return False

  def __gt__(self, other: ctstruct) -> bool: # >
    for fld in self._fields_:
      if not isinstance(fld, tuple(cts)):
        if str(self._fields_[1]) > str(other._fields_[1]): return True
      else:
        if getattr(self, fld[0]) > getattr(other, fld[0]): return True
    return False

  def __ge__(self, other: ctstruct) -> bool: # >=
    for fld in self._fields_:
      if not isinstance(fld, tuple(cts)):
        if str(self._fields_[1]) >= str(other._fields_[1]): return True
      else:
        if getattr(self, fld[0]) >= getattr(other, fld[0]): return True
    return False

class dtype(ctstruct):
  _fields_: List[Any] = []
  def __init__(self, dtype: Type[Any]=ctypes.c_uint8) -> None:
    self._fields_ = [dtype, dtype]
    self.type = self._fields_[1]
  def __repr__(self) -> str: return repr(types[self._fields_[1]]).replace("'", "") if (self._fields_[1] and str(self._fields_[1]) not in types) else str(self._fields_[1])
  def __call__(self, x: List[Any]) -> Any: return self._fields_[1]
  def value(self) -> Any: return self._fields_[0] if not isinstance(tuple(self._fields_), tuple(cts)) else self._fields_[1]

# types
float16 = dtype(dtype=ctypes.c_float)
float32 = dtype(dtype=ctypes.c_float)
float64 = dtype(dtype=ctypes.c_double)
int8 = dtype(dtype=ctypes.c_int8)
int16 = dtype(dtype=ctypes.c_int16)
int32 = dtype(dtype=ctypes.c_int32)
int64 = dtype(dtype=ctypes.c_int64)
uint8 = dtype(dtype=ctypes.c_uint8)
uint16 = dtype(dtype=ctypes.c_uint16)
uint32 = dtype(dtype=ctypes.c_uint32)
uint64 = dtype(dtype=ctypes.c_uint64)
bool_ = dtype(dtype=ctypes.c_bool)
types = {'float16': 'ctypes.c_float', 'float32': 'ctypes.c_float', 'float64': 'ctypes.c_double', 'int8': 'ctypes.c_int8', 'int16': 'ctypes.c_int16', 'int32': 'ctypes.c_int',
  'int64': 'ctypes.c_int64', 'uint8': 'ctypes.c_uint8', 'uint16': 'ctypes.c_uint16', 'uint32': 'ctypes.c_uint32', 'uint64': 'ctypes.c_uint64', 'bool_': 'ctypes.c_bool',
  ctypes.c_float: 'float16', ctypes.c_float: 'float32', ctypes.c_double: 'float64', ctypes.c_int8: 'int8', ctypes.c_int16: 'int16', ctypes.c_int32: 'int32', ctypes.c_int64: 'int64',
  ctypes.c_uint8: 'uint8', ctypes.c_uint16: 'uint16', ctypes.c_uint32: 'uint32', ctypes.c_int: 'int32', ctypes.c_uint64: 'uint64', ctypes.c_bool: 'bool_', ctypes.c_ubyte: 'uint8',
}

# constants
inf = math.inf
nan = math.nan
