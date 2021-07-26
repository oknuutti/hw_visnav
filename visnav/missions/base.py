from abc import ABC


class Mission(ABC):
    def __init__(self, video_path, data_path=None, video_toff=0, first_frame=None, last_frame=None, live=False):
        self.video_path = video_path
        self.data_path = data_path
        self.video_toff = video_toff
        self.first_frame = first_frame
        self.last_frame = last_frame
        self.data, self.time0, self.coord0 = [None]*3 if live else self.init_data()
        self.cam = self.init_cam()
        self.odo = self.init_odo()

    def init_cam(self):
        raise NotImplemented()

    def init_odo(self):
        raise NotImplemented()

    def init_data(self):
        raise NotImplemented()

