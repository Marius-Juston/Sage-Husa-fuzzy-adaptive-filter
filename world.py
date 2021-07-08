from abc import ABC, abstractmethod
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from csv_reader import CSVReader
from robot import UKFRobot, UKFROSRobot
from ukf.datapoint import DataType
from ukf.state import UKFState

np.random.seed(421)


def angle_diff(angle1, angle2):
    diff = (angle2 - angle1 + np.pi) % (2 * np.pi) - np.pi

    diff[diff < -np.pi] += 2 * np.pi

    return diff


class BaseWorld(ABC):
    @abstractmethod
    def calculate_rsme(self, robot_id):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def plot(self, ranges=False, offset_sensor=False, large_errors=False):
        pass


class World(BaseWorld):
    def __init__(self, dt=1 / 60, d_std=.1, large_d_p=.075, large_d_std=2) -> None:
        super().__init__()

        self.t = 0

        self.dt = dt
        self.large_d_std = large_d_std
        self.large_d_p = large_d_p
        self.d_std = d_std
        self.uwb_sensors = []
        self.large_errors = []
        self.robots: List[Dict[str, Any]] = []

    def add_uwb(self, pose):
        pose = np.asarray(pose, dtype=float)

        self.uwb_sensors.append(pose)

    def add_robot(self, robot):
        self.robots.append({
            'robot': robot,
            'w': [],
            'r': [],
            'e': [],
            'e_t': [],
            'w_t': []
        })

    def calculate_rsme(self, robot_id):
        robot_data = self.robots[robot_id]
        actual = np.asarray(robot_data['w'], dtype=float)
        actual_t = np.asarray(robot_data['w_t'], dtype=float)
        expected = np.asarray(robot_data['e'], dtype=float)
        expected_t = np.asarray(robot_data['e_t'], dtype=float)

        rsme = np.mean((actual - expected) ** 2, axis=0)
        rsme_t = np.mean(angle_diff(actual_t, expected_t) ** 2)

        rsme = np.append(rsme, rsme_t)

        return np.sqrt(rsme)

    def reset(self):
        self.t = 0

        self.uwb_sensors = []
        self.robots = []

        self.world.clear()
        self.ranges.clear()
        self.angles.clear()

    def step(self):
        sensor = self.uwb_sensors.pop(0)
        self.uwb_sensors.append(sensor)

        for robot_data in self.robots:
            r = robot_data['robot']

            r.move(self.dt)

            if isinstance(r, UKFRobot):
                pose = r.get_sensor_pose()
                robot_data['w'].append(r.get_pose())
            else:
                pose = r.get_pose()
                robot_data['w'].append(pose)
            actual_t = r.get_heading()

            d = np.linalg.norm(np.array(sensor) - np.array(pose))

            will_large = np.random.uniform(0, 1)

            if will_large < self.large_d_p:
                noise = np.random.normal(0, self.large_d_std)
                self.large_errors.append(r.get_pose())
            else:
                noise = np.random.normal(0, self.d_std)

            d += noise

            estimated_pose, estimated_t = r.localize(d, sensor, self)

            # robot_data['r'].append(d)
            robot_data['r'].append(noise)
            robot_data['e'].append(estimated_pose)
            robot_data['e_t'].append(estimated_t)
            robot_data['w_t'].append(actual_t)

        self.t += self.dt

    def plot(self, ranges=False, offset_sensor=False, large_errors=False):
        f: Figure = plt.gcf()

        a = plt.figure().gca()

        if ranges:
            self.world, self.ranges, self.angles = f.subplots(ncols=3)
        else:
            self.world, self.angles = f.subplots(ncols=2)
        self.world.set_aspect('equal')

        for pose in self.uwb_sensors:
            self.world.scatter(*pose[:2], label='anchor')

        for i, r in enumerate(self.robots):
            estimated = np.asarray(r['e'])
            actual = np.asarray(r['w'])
            estimated_t = np.asarray(r['e_t'])
            actual_t = np.asarray(r['w_t'])
            range = r['r']

            x = np.arange(0, len(estimated)) * self.dt

            if len(self.large_errors) > 0 and large_errors:
                large_errors = np.asarray(self.large_errors)
                self.world.scatter(large_errors[:, 0], large_errors[:, 1], zorder=10)
            self.world.plot(estimated[:, 0].flatten(), estimated[:, 1].flatten(), label=f'actual {i}')
            self.world.plot(actual[:, 0].flatten(), actual[:, 1].flatten(), label=f'expected {i}')
            self.angles.plot(x, estimated_t, label=f'expected {i}')
            self.angles.plot(x, actual_t, label=f'actual {i}')
            a.plot(x, estimated_t, label=f'expected {i}')
            a.plot(x, actual_t, label=f'actual {i}')

            if offset_sensor:
                if isinstance(r['robot'], UKFRobot):
                    s_p = np.array(r['robot'].sensor_pose)

                    self.world.plot(s_p[:, 0], s_p[:, 1])
                #     anchor_pose = np.array(r['robot'].ukf.measurement_predictor.data)
                #     i = 0
                #
                #     while i < 17:
                #         self.world.plot(anchor_pose[:, 0, i], anchor_pose[:, 1, i])
                #         i += 1
            if ranges:
                self.ranges.scatter(x, range, label=f'robot {i}')

        # self.world.legend()
        if ranges:
            self.ranges.legend()


class ROSWorld(BaseWorld):
    cache = None

    def __init__(self, csv_file) -> None:
        super().__init__()

        if ROSWorld.cache is None:
            self.csv = CSVReader(csv_file, use_ground_truth=True)
            self.csv.process()
            ROSWorld.cache = self.csv
        else:
            self.csv = ROSWorld.cache
        self.index = 0

        self.robot = None
        self.estimated_pose = []
        self.estimated_heading = []
        self.ground_truths = []
        self.times = []
        self.ground_truth_index = 0

        self.estimated_state = []
        self.ground_truths = []

    def interpolate_pose(self, i_pose, i_t, f_pose, f_t, x_t):
        o = i_pose + (x_t - i_t) * (f_pose - i_pose) / (f_t - i_t)

        return o

    def set_robot(self, robot):
        self.robot = robot

    def calculate_rsme(self, robot_id):
        # return np.sum(np.linalg.norm(np.array(self.estimated_pose) - np.array(self.ground_truths), axis=1))
        # np.linalg.norm(np.array(self.estimated_pose) - np.array(self.ground_truths), axis=1)

        estimated_pose = np.array(self.estimated_pose)
        ground_truths = np.array(self.ground_truths)

        rsme = np.mean((estimated_pose - ground_truths) ** 2, axis=0)

        rsme[UKFState.YAW] = np.mean(angle_diff(estimated_pose[:, UKFState.YAW], ground_truths[:, UKFState.YAW]) ** 2)

        return np.sqrt(rsme)
        # return rsme

    def reset(self):
        pass

    def empty(self):
        return self.index >= len(self.csv.sequential_data)

    def step(self):
        if self.robot is not None:
            data_point = self.csv.sequential_data[self.index]

            t = data_point.timestamp

            gt = self.get_closest_ground_truth(t)
            self.ground_truths.append(gt)

            self.times.append(t)
            self.robot.localize(data_point)
            # self.estimated_pose.append(self.robot.get_pose())
            self.estimated_heading.append(self.robot.get_heading())

            self.estimated_pose.append(self.robot.ukf.x)

        self.index += 1

    def plot(self, ranges=False, offset_sensor=False, large_errors=False, interpolation=False):
        f: Figure = plt.figure(0)

        if ranges:
            self.world, self.ranges, self.angles = f.subplots(ncols=3)
        else:
            self.world, self.angles = f.subplots(ncols=2)
        self.world.set_aspect('equal')

        ground_truth = np.array([i.measurement_data for i in self.csv.sensor_data[DataType.GROUND_TRUTH]])
        ground_truth_time = np.array([i.timestamp for i in self.csv.sensor_data[DataType.GROUND_TRUTH]])
        esimtated = np.array(self.estimated_pose)

        if interpolation:
            inter_gts = np.array(self.ground_truths)
            self.world.plot(inter_gts[:, 0], inter_gts[:, 1], zorder=-1, linewidth=4)

        self.world.plot(ground_truth[:, 0], ground_truth[:, 1])
        self.world.plot(esimtated[:, 0], esimtated[:, 1])

        self.angles.plot(ground_truth_time, ground_truth[:, UKFState.YAW])
        self.angles.plot(self.times, self.estimated_heading, c='b', zorder=-1, linewidth=2.5)
        # s = np.array(a)
        # self.angles.plot(s[:, 0], s[:, 1])

        # plt.show()

        x = ['x', 'y', 'z', 'v', '$\psi$', '$\dot{\psi}$']

        for i in range(esimtated.shape[-1]):
            fig: Figure = plt.figure(i + 1)
            fig.suptitle(x[i])
            a = fig.gca()
            a.plot(self.times, esimtated[:, i], label='Estimated')
            a.plot(ground_truth_time, ground_truth[:, i], label='Actual')
            a.legend()
        plt.show()

    def get_closest_ground_truth(self, t):
        gts = self.csv.sensor_data[DataType.GROUND_TRUTH]

        while self.ground_truth_index >= 0 and t <= gts[self.ground_truth_index].timestamp:
            self.ground_truth_index -= 1

        while self.ground_truth_index < len(gts) - 2 and not (gts[self.ground_truth_index].timestamp <= t and t <= gts[
            self.ground_truth_index + 1].timestamp):
            self.ground_truth_index += 1

        o = self.interpolate_pose(gts[self.ground_truth_index].measurement_data[:6],
                                  gts[self.ground_truth_index].timestamp,
                                  gts[self.ground_truth_index + 1].measurement_data[:6],
                                  gts[self.ground_truth_index + 1].timestamp, t)

        return o


def ukf_test_world():
    w = World(dt=.1, large_d_p=0.0)
    # w.add_robot(RandomRobot(v=1., w=.2))

    w.add_uwb((60, 0, 0))
    w.add_uwb((-10, 0, 0))
    w.add_uwb((0, -60, 0))
    w.add_uwb((0, 10, 0))
    # w.add_uwb((3, -5, 0))
    # w.add_uwb((20, 10, 0))
    # x = [0.05, 0.05, 1.040286859872973, 3.5881729155829003, 0.042798706624045814, 2.3200423063247566]
    # x = [0.10847719179050998, 0.7631467014140771, 2.7323491946570404, 0.9799270449756307, 2.1627110708845776, -1.8284493304109404]
    # x = [0.0900102578988983, 0.3049731348367063, 2.463590551872387, 1.2536250456200835, 0, -5.798615125929821]
    # x = [0.06336593385591043, 0.06415036900110617, 2.3417757588529806, 1.0827531351971345, 1, -0.09536437601070347]
    # x = [0.0656735202138201, 0.005, 2.1546593167987678, 0.7624456375027308, 1, 0.9248206058828421]
    # Best position based
    # x = [0.06440231674683623, 0.005, 2.11960504430087, 1.1198198133625668, 1, 2.83947027741379]
    # x = [0.09275029262337292, 0.44998668126287167, 1.632607406752226, 0.001, 0, 6]
    # x = [0.05583089055343559, 2.3162773302242474, 0.8666377950612036, 0.001, 1, -3]
    # x = [0.1000479619463182, 4.309831513468891, 0.6131229717593857, 0.7230945231505106, 1.9240049222228839, -0.08891244961775335]
    # x = [0.09323979462800462, 2.325254927326243, 0.7411337649150395, 0.024489667725622666, 1, -3.5723911416132106]
    # 0.011702153331107044
    x = [0.20796177443716282, 2.424418915956577, 0.5860493507682885, 0.7874421557351899, -0.6146625234589184,
         -4.405076722968345]
    # Best position and heading based
    # x = [.10666510053525305, 0.24935980204404887, 1.9923906279320838, 0.7098003276876328, 1, 0.9785413756248991]

    w.add_robot(UKFRobot(v=1., w=.2, uwb_std=x[0], speed_noise_std=x[1], yaw_rate_noise_std=x[2], alpha=x[3], beta=x[4],
                         k=x[5]))
    # w.add_robot(UKFRobot(v=1., w=.2))
    for i in range(2000):
        try:
            for r in w.robots:
                if i % 100 == 0:
                    c = np.random.choice((1, 2, 3, 4))

                    if c == 1:
                        r['robot'].w = 2 * r['robot'].w
                    if c == 2:
                        r['robot'].w = -r['robot'].w
                    if c == 3:
                        r['robot'].w = .5 * r['robot'].w
                    if c == 4:
                        # pass
                        r['robot'].v = -r['robot'].v

            print("Step:", i)
            w.step()
        except np.linalg.LinAlgError:
            print("LinAlgError")
            break

    print(w.robots[0]['robot'].get_pose())

    for i, r in enumerate(w.robots):
        rsme = w.calculate_rsme(i)
        print(i, "RSME", rsme, np.sum(rsme[:3]))

    w.plot(ranges=False, offset_sensor=False, large_errors=True)

    plt.show()


def ros_world():

    init_pose = w.csv.sensor_data[DataType.GROUND_TRUTH][0]

    # w.set_robot(
    #     UKFROSRobot(init_pose.measurement_data[:3], init_pose.timestamp, t=init_pose.measurement_data[UKFState.YAW], P=
    #     np.diag([0.0001, 0.0001, 0.0001, .0001, 0.0001, 0.0001])))

    # 0.47603732478383814
    x = [1.0001, 3.9001, 4.9001, 1, 0, None, 0.0001, 0.0001, 0.0001,
         2.0001, 0.0001, 0.0001, 11.0, 14.0001, 20.9001, 1.0001, 0.0001, 0.0001, 1]
    # x = [0.3005500739580095, 1.7456795243473833, 1.2537087820968702, 0.9756936851785581, 0.12282062791567361, -2.685579681516146, 1.7811686059759613, 8.135101518427398, 0.0001, 0.0001, 9.679152133012991, 6.493196017200628]
    # x = [.1005500739580095, 0.27456795243473833, 0.537087820968702, 0.9756936851785581, 0.12282062791567361, -2.685579681516146, 1.7811686059759613, 8.135101518427398, 0.0001, 0.0001, 9.679152133012991, 6.493196017200628]
    # x = [0.5356205657134541, 0.18109875383894544, 0.3323663017573703, 0.001, 2.0, -4.899251280615886, 2.3463439365430467, 0.0001, 0.0001, 2.8112198335402963, 0.0001, 9.20133215026015]
    # x = [3.0, 0.688804761738878, 2.4316845626457964, 3.0, -0.9976005928055613, 0.0, 10.0, 7.474000642513913,
    #      16.74751457478579, 3.986383868394205, 0.9981002081983105, 10.0, 0.01, 0.01, 0.8348157603174587,
    #      3.1186254198285264, 2.8082493403438566, 1.0723748068270464, .01]
    x = [2.596885350765864, 2.0721395287970763, 1.315604098506842, 2.0368880179016307, 0.6786608370724769,
         -0.9667802487818076, 1.1703781655446577, 10.0, 0.0001, 0.0001, 1.0193177487189773, 9.572643654938378,
         15.150962296254791, 13.16177619327417, 13.527487408284795, 0.36311133022569103, 2.9299157429679155,
         1.5223747167468766, 2.187179836412636]

    w.set_robot(
        UKFROSRobot(init_pose.measurement_data[:3], init_pose.timestamp, t=init_pose.measurement_data[UKFState.YAW],
                    uwb_std=x[0],
                    speed_noise_std=x[1],
                    yaw_rate_noise_std=x[2],
                    alpha=x[3],
                    beta=x[4],
                    k=x[5],
                    P=np.diag([x[6], x[7], x[8], x[9], x[10], x[11]]),
                    odometry_std=x[12:18],
                    imu_std=(x[18],)
                    ))

    try:
        while not w.empty():
            print("Step:", w.index)
            w.step()
    except np.linalg.LinAlgError:
        pass

    rsme = w.calculate_rsme(0)
    # print("RSME", rsme, np.sum(rsme[:3]))
    print("RSME", rsme, np.sum(rsme))

    w.plot(ranges=False, offset_sensor=False, large_errors=True)

    plt.show()


if __name__ == '__main__':
    # ukf_test_world()
    ros_world()
