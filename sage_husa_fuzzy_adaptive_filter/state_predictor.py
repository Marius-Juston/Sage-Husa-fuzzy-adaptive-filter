import numpy as np


class StatePredictor(object):

    def __init__(self) -> None:
        super().__init__()

        self.x = None
        self.P = None
        self.I_3 = np.identity(3)

    def calcualte_F(self, dt, C, am, wm):
        I = self.I_3
        I_dt = I * dt

        ab = self.x[10:13]
        wb = self.x[13:16]

    def process(self, dt, imu):
        C = np.identity(3)
