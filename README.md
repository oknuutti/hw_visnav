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
pip install moderngl==5.5.4 moderngl-ext-obj==1.0.0
```

## Run a test
Download test data from [here](https://drive.google.com/file/d/1JepyAQa2jZpCBJPfJjeAh53xw33QPLMa/view?usp=sharing) to `tests/data`.

Then run:

```
python -m unittest tests/odometry.py
```

or to test rendering with profiling, run:
```
xvfb-run --server-args="-screen 0 1024x768x24" python -m cProfile -o profile.prof -m tests.render
```

or to test synthetic photometric feature algo with config set "1", run:
```
xvfb-run --server-args="-screen 0 1024x768x24" python -m cProfile -o profile1.prof tests/splnav.py 1
```

## TODO:
- include inertial measurements

