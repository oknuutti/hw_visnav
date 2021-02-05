from abc import ABC


class Mission(ABC):
    def __init__(self, datapath, first_frame=None, last_frame=None, live=False):
        self.datapath = datapath
        self.first_frame = first_frame
        self.last_frame = last_frame
        self.cam = self.init_cam()
        self.odo = self.init_odo()
        self.data, self.time0, self.prior = [None]*3 if live else self.init_data()

    def init_cam(self):
        raise NotImplemented()

    def init_odo(self):
        raise NotImplemented()

    def init_data(self):
        raise NotImplemented()

