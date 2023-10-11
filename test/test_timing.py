import numpy as np
import lightnum.lightnum as lp
from datetime import datetime

def test_timing():
  start_time = datetime.now()
  for _ in range(10000): np.mod([1, 2, 3, 4], [1, 2, 6, 4])
  time_elapsed = datetime.now() - start_time
  print('NP Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

  start_time = datetime.now()
  for _ in range(10000): lp.mod([1, 2, 3, 4], [1, 2, 6, 4])
  time_elapsed = datetime.now() - start_time
  print('LP Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
