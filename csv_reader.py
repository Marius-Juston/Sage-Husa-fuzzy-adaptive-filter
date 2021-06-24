from typing import Dict, Callable, List, Any

import numpy as np

from ukf.datapoint import DataType, DataPoint


class CSVReader:

    def __init__(self, csv_file) -> None:
        super().__init__()
        self.processor = {
            DataType.UWB: self.process_uwb,
            DataType.IMU: self.process_imu,
            DataType.GROUND_TRUTH: self.process_odom,
            DataType.ODOMETRY: self.process_odom
        }

        self.csv_file = csv_file

        self.ground_truth = []
        self.sequential_data = []
        self.sensor_data = dict()

    def process(self):
        self.processor: Dict[int, Callable[[int, List[str]], Any]]

        with open(self.csv_file) as file:
            for line in file.readlines():
                line_data = line.split(',')
                id, t = map(int, line_data[:2])

                if id not in self.sensor_data:
                    self.sensor_data[id] = []

                processed = self.processor[id](t, line_data[2:])

                if id != DataType.GROUND_TRUTH:
                    self.sequential_data.append(processed)

                self.sensor_data[id].append(processed)

                print(processed)

    def process_imu(self, t, line_data):
        data = DataPoint(DataType.IMU, np.asarray(tuple(map(float, line_data))), t)

        return data

    def process_uwb(self, t, line_data):
        # anchor_distance,
        # anchor_pose[0], anchor_pose[1], anchor_pose[2],
        # tag[0], tag[1], tag[2]
        line_data = tuple(map(float, line_data))

        d = line_data[0]
        anchor = line_data[1:4]
        tag = line_data[4:]

        data = DataPoint(DataType.UWB, d, t, extra={
            "anchor": anchor,
            'sensor_offset': tag
        })

        return data

    def process_odom(self, t, line_data):
        # id, t, px, py, pz, v, theta, theta_yaw, msg.pose.pose.orientation.x,
        # msg.pose.pose.orientation.y,
        # msg.pose.pose.orientation.z,
        # msg.pose.pose.orientation.w
        data = DataPoint(DataType.ODOMETRY, np.array(tuple(map(float, line_data))), t)

        return data


if __name__ == '__main__':
    reader = CSVReader('out.csv')
    print("Processing")
    reader.process()
    print("Done")
