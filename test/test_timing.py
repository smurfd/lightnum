import numpy as np
import lightnum.lightnum as lp
from datetime import datetime

def test_timing():
  start_time = datetime.now()
  for _ in range(1000000): np.mod([1, 2, 3, 4], [1, 2, 6, 4])
  time_elapsed1 = datetime.now() - start_time
  print('NP Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed1))

  start_time = datetime.now()
  for _ in range(1000000): lp.mod([1, 2, 3, 4], [1, 2, 6, 4])
  time_elapsed2 = datetime.now() - start_time
  print('LP Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed2))
  assert(time_elapsed2 < time_elapsed1)

  start_time = datetime.now()
  for _ in range(1000000): np.sqrt([1, 2, 3, 4])
  time_elapsed1 = datetime.now() - start_time
  print('NP Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed1))

  start_time = datetime.now()
  for _ in range(1000000): lp.sqrt([1, 2, 3, 4])
  time_elapsed2 = datetime.now() - start_time
  print('LP Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed2))
  assert(time_elapsed2 < time_elapsed1)
  # Faster than numpy!!!
  # NP Time elapsed (hh:mm:ss.ms) 0:00:01.323887
  # LP Time elapsed (hh:mm:ss.ms) 0:00:01.178521
test_timing()
print("OK!")
