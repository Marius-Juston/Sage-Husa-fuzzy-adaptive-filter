from abc import ABC, abstractmethod

import numpy as np

from ukf.datapoint import DataType, DataPoint
from ukf.fusion_ukf import FusionUKF
from ukf.state import UKFState
from util import euler_from_quaternion


class Robot(ABC):

    def __init__(self, pose=None, t: float = 0, v: float = 0, w: float = 0) -> None:
        super().__init__()

        if pose is None:
            self.pose = np.zeros(3)
        else:
            self.pose = pose

        self.v = v
        self.w = w
        self.t = t

    def set_v(self, v):
        self.v = v

    def set_w(self, w):
        self.w = w

    def get_pose(self):
        return self.pose

    def move(self, dt):
        px = self.pose[0]
        py = self.pose[1]
        pz = self.pose[2]
        speed = self.v
        yaw = self.t
        yaw_rate = self.w

        # PREDICT NEXT STEP USING CTRV Model

        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        # Velocity change
        d_yaw = yaw_rate * dt
        d_speed = speed * dt

        # Predicted speed = constant speed + acceleration noise
        p_speed = speed

        # Predicted yaw
        p_yaw = yaw + d_yaw

        # Predicted yaw rate
        p_yaw_rate = yaw_rate

        if yaw_rate <= 0.0001:
            p_px = px + d_speed * cos_yaw
            p_py = py + d_speed * sin_yaw

        else:
            k = speed / yaw_rate
            theta = yaw + d_yaw
            p_px = px + k * (np.sin(theta) - sin_yaw)
            p_py = py + k * (cos_yaw - np.cos(theta))

        self.pose = np.array([p_px, p_py, pz])
        self.v = p_speed
        self.t = (p_yaw + np.pi) % (2 * np.pi) - np.pi
        self.w = p_yaw_rate

    def get_heading(self):
        return self.t

    @abstractmethod
    def localize(self, d, anchor_pose, w):
        pass


class RandomRobot(Robot):

    def localize(self, d, anchor_pose, w):
        return self.pose + np.random.normal(0, .05, 3), self.t + np.random.normal(0, .05)


class UKFRobot(Robot):

    def __init__(self, init_pose=None, pose=None, t=0, v=0, w=0, uwb_std=.1, speed_noise_std=.1, yaw_rate_noise_std=2.5,
                 alpha=1., beta=0., k=None) -> None:
        super().__init__(pose, t, v, w)

        # p = [1.0001, 11.0, 14.0001, 20.9001, 1.0001, 0.0001, 0.0001, 3.9001, 4.9001, 1.0, 0, 0.0001, 0.0001, 0.0001,             2.0001, 0.0001, 0.0001]

        if init_pose is None:
            self.init_pose = self.pose + np.random.normal(0, .1, 3)
        else:
            self.init_pose = init_pose

        sensor_std = {
            DataType.UWB: {
                'std': [uwb_std],
                'nz': 1
            }
        }

        self.sensor_pose = []

        self.ukf = FusionUKF(sensor_std, speed_noise_std=speed_noise_std, yaw_rate_noise_std=yaw_rate_noise_std,
                             alpha=alpha, beta=beta, k=k)
        self.ukf.initialize(np.array([*self.init_pose, v, t, w]), np.identity(6) / 100, 0)

        self.sensor = np.array([0, -.162, .184])

    def rotation_matrix(self, angle):
        s = np.sin(angle)
        c = np.cos(angle)

        return ((c, -s, 0),
                (s, c, 0),
                (0, 0, 1))

    def get_pose(self):
        return self.pose

    def get_sensor_pose(self):
        rot = self.rotation_matrix(self.t)

        p = self.pose + rot @ self.sensor
        self.sensor_pose.append(p)

        return p

    def localize(self, d, anchor_pose, w):
        data = DataPoint(DataType.UWB, d, w.t * 1e9, extra={
            "anchor": anchor_pose,
            'sensor_offset': self.sensor
        })

        # print(self.ukf.x[:3])
        self.ukf.update(data)

        # print(np.degrees(self.ukf.x[4]))

        return self.ukf.x[:UKFState.Z + 1], self.ukf.x[UKFState.YAW]


class ROSRobot(ABC):

    def __init__(self, pose, t) -> None:
        super().__init__()
        self.pose = pose
        self.t = t

    @abstractmethod
    def localize(self, data):
        pass

    def get_pose(self):
        return self.pose

    def get_heading(self):
        return self.t


class UKFROSRobot(ROSRobot):
    def __init__(self, init_pose, inital_time, t=0, v=0, w=0, uwb_std=1.0001,
                 odometry_std=(11.0, 14.0001, 20.9001, 1.0001, 0.0001, 0.0001), imu_std=(0.01,), speed_noise_std=.0101,
                 yaw_rate_noise_std=.01001, alpha=1, beta=0, k=None, P=None, sensor_used=None) -> None:
        super().__init__(init_pose, t)

        if sensor_used is None:
            self.sensor_used = [DataType.IMU, DataType.ODOMETRY, DataType.UWB]
        else:
            self.sensor_used = sensor_used

        self.init_pose = init_pose

        sensor_std = {
            DataType.UWB: {
                'std': [uwb_std],
                'nz': 1
            },
            DataType.ODOMETRY: {
                'std': odometry_std,
                'nz': 6
            },
            DataType.IMU: {
                'std': imu_std,
                'nz': 1
            }
        }

        self.sensor_pose = []

        if P is None:
            P = np.identity(6)

        self.ukf = FusionUKF(sensor_std, speed_noise_std=speed_noise_std, yaw_rate_noise_std=yaw_rate_noise_std,
                             alpha=alpha, beta=beta, k=k)
        self.ukf.initialize(np.array([*self.init_pose, v, t, w]), P, inital_time)

    def localize(self, data: DataPoint):
        if data.data_type == DataType.ODOMETRY:
            data.measurement_data = data.measurement_data[:6]
        elif data.data_type == DataType.IMU and data.measurement_data.size > 1:
            yaw = euler_from_quaternion(data.measurement_data[:4])[2]

            data.measurement_data = np.array([yaw])

        if data.data_type in self.sensor_used:
            self.ukf.update(data)
        else:
            self.ukf.predict(data)

        self.pose = self.ukf.x[:UKFState.Z + 1]
        self.t = self.ukf.x[UKFState.YAW]

        return self.pose, self.t
