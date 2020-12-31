import time

from sense_hat.sense_hat import SenseHat


class OwnSenseHAT(SenseHat):
    def __init__(self, *args, **kwargs):
        super(OwnSenseHAT, self).__init__(*args, **kwargs)
        self.raw_data = None

    def read_raw_data(self):
        if self._read_imu():
            self.raw_data = self._imu.getIMUData()
            if self.raw_data['compassValid']:
                self._last_compass_raw = self.raw_data['compass']
            if self.raw_data['gyroValid']:
                self._last_gyro_raw = self.raw_data['gyro']
            if self.raw_data['accelValid']:
                self._last_accel_raw = self.raw_data['accel']
            return True
        return False

    def get_compass_raw(self):
        return self._last_compass_raw

    def get_gyroscope_raw(self):
        return self._last_gyro_raw

    def get_accelerometer_raw(self):
        return self._last_accel_raw

    def _read_imu(self):
        """
        Internal. Tries to read the IMU sensor three times before giving up
        """

        self._init_imu()  # Ensure imu is initialised

        attempts = 0
        success = False

        while not success and attempts < 3:
            success = self._imu.IMURead()
            if success:
                break
            attempts += 1
            time.sleep(self._imu_poll_interval)

        return success
