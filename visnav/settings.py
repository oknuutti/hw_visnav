
import os

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
DATA_DIR = os.path.join(BASE_DIR, 'data')

USE_ICRS = True  # if true, use barycentric equatorial system, else heliocentric ecliptic

# render this size synthetic images on the s/c, height is proportional to cam height/width
VIEW_WIDTH = 512

DEBUG = 1
BATCH_MODE = 1  # useless variable but referred to at some places
