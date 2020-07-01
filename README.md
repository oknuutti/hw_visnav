## Installation / dependencies

`odometry.py` should work with:
```
pip install numpy numpy-quaternion opencv-contrib-python scipy
```

Can skip `scipy` if it's too difficult, should try to use [ceres-solver](https://anaconda.org/conda-forge/ceres-solver) then.
Maybe with python bindings from [here](https://github.com/Edwinem/ceres_python_bindings)?

Later on when trying to get other algorithms to work:
```
apt-get install libgl1-mesa-dev libx11-dev
pip install numba astropy moderngl
```

## Run a test
Download test data from [here](https://drive.google.com/file/d/1JepyAQa2jZpCBJPfJjeAh53xw33QPLMa/view?usp=sharing) to `tests/data`.

Then run:

```
python -m unittest tests/odometry.py
```

## TODO:
- fix odometry.py, for some reason there's high orientation errors


