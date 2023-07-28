### light numpy (lightnum)

The idea is to be able to switch (for some functions atleast):

`import numpy as np` with `import lightnum.lightnum as np`

No dependencies, Python >= 3.8

### Install & install testing
```
python3 -m pip install -e '.[testing, linting]'
```
### Test
```
python3 -m pytest
```
### Use in your project
(from your projects root path)
```
python3 /path/to/lightnum/replace_np.py
```
