from typing import Dict, Callable, List, Any

from ukf.datapoint import DataType


class CSVReader:

    def __init__(self, csv_file) -> None:
        super().__init__()
        self.processor = {
            DataType.UWB: self.process_uwb,
            DataType.IMU: self.process_imu,
            DataType.GROUND_TRUTH: self.process_ground_truth,
            DataType.ODOMETRY: self.process_odom
        }

        self.csv_file = csv_file

        self.ground_truth = []
        self.sequential_data = []
        self.sensor_data = dict()
        self.sensor_data.setdefault([])

    def process(self):
        self.processor: Dict[int, Callable[[List[str]], Any]]

        with open(self.csv_file) as file:
            for line in file.readlines():
                line_data = line.split(',')
                id, t = map(int, line_data[:2])

                processed = self.processor[id](line_data[2:])

                self.sequential_data.append(processed)
                self.sensor_data[id].append(processed)

    def process_imu(self, line_data):
        line_data = tuple(map(float, line_data))

        orientation = line_data[:4]
        angl_vel = line_data[4:7]
        lin_acc = line_data[7:]

        return orientation, angl_vel, lin_acc

    def process_uwb(self, line_data):
        pass

    def process_odom(self, line_data):
        # id, t, px, py, pz, v, theta, theta_yaw, msg.pose.pose.orientation.x,
        # msg.pose.pose.orientation.y,
        # msg.pose.pose.orientation.z,
        # msg.pose.pose.orientation.w
        pass

    def process_ground_truth(self, line_data):
        pass


if __name__ == '__main__':
    reader = CSVReader('out.csv')
    print("Processing")
    reader.process()
    print("Done")
