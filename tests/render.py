import os
import math
import logging
import warnings
warnings.filterwarnings("ignore", module='quaternion', lineno=21)

import numpy as np
import cv2
import sys

from visnav.render.render import RenderEngine
from visnav.algo import tools
from visnav.missions.rosetta import RosettaSystemModel


def main(outfile='render-test.jpg'):
    logger = logging.getLogger(__name__)
    logger.info('Setting up renderer and loading the 3d model...')

    sm = RosettaSystemModel(hi_res_shape_model=False, skip_obj_load=True)

    re = RenderEngine(sm.cam.width, sm.cam.height, antialias_samples=0)
    re.set_frustum(sm.cam.x_fov, sm.cam.y_fov, 0.1 * sm.min_distance, 1.1 * sm.max_distance)
    obj_path = os.path.join(os.path.dirname(__file__), 'data', '67p-16k.obj')
    obj_idx = re.load_object(obj_path)

    pos = np.array([0, 0, -sm.min_med_distance * 1])
    q = tools.angleaxis_to_q((math.radians(-20), 0, 1, 0))
    light_v = np.array([1, 0, -1]) / math.sqrt(2)

    logger.info('Start rendering...')
    with tools.Stopwatch() as time:
        img = re.render(obj_idx, pos, q, light_v, gamma=1.8, get_depth=False, shadows=True, textures=True,
                        reflection=RenderEngine.REFLMOD_HAPKE)

    logger.info('Rendered one image in %f secs' % time.elapsed)
    ok = cv2.imwrite(outfile, img)
    if ok:
        logger.info('Result saved in file %s' % outfile)
    else:
        logger.warning('Failed to save image in file %s' % outfile)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1] if len(sys.argv) > 1 else 'render-test.jpg')
