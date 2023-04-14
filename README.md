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
python -mr unittest tests/odometry.py
```

or to test rendering with profiling, run:
```
xvfb-run --server-args="-screen 0 1024x768x24" python -mr cProfile -o profile.prof -mr tests.render
```

or to test synthetic photometric feature algo with config set "1", run:
```
xvfb-run --server-args="-screen 0 1024x768x24" python -mr cProfile -o profile1.prof tests/splnav.py 1
```

## For processing drone videos:
```
conda create -n drone -c conda-forge python=3.10 pip numpy numba quaternion scipy opencv python-dateutil tqdm \
                                     matplotlib
conda activate drone
pip install opencv-contrib-python
pip install pygeodesy
pip install --no-deps kapture
```

### If need telemetry parser for Nokia data:
```
mkdir 3rdparty
cd 3rdparty
git clone git@version.aalto.fi:jkinnari/nokia-multico-dataset-parser.git -b okn-work nokia-ds
cd ..
pip install --no-deps -e 3rdparty/nokia-ds
```

## TODO:
- include inertial measurements

