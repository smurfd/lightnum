### light numpy (lightnum)

The idea is to be able to switch (for some functions atleast):

`import numpy as np` with `import lightnum.lightnum as np`

No dependencies, Python >= 3.8 (for testing, numpy, since its compared towards numpy)

### Install & install testing
```
python3 -m pip install -e '.[testing, linting]' --user
```
### Test
```
python3 -m pytest lightnum/test
```
### Use in your project
(After you have installed lightnum with `pip`. From your projects root path, run)
```
python3 /path/to/lightnum/lightnum/scripts/replace_np.py
```
### Create a portable version
```
python3 lightnum/scripts/create_portable.py
```
Then copy `lightnum.py` to your project folder, and use with:
```
import lightnum
```
If you plan to put the portable file in a folder like `stuff`
search for `import lightnum` in the newly created `lightnum.py` and replace it with `import stuff.lightnum`
If you want to run the lightnum tests with the portable version, edit the tests and replace `import lightnum.lightnum` with the above
