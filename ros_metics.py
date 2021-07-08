import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from csv_reader import CSVReader
from ukf.datapoint import DataType, DataPoint


class ROSMetrics:

    def __init__(self, csv_file) -> None:
        super().__init__()

        self.csv = CSVReader(csv_file)
        self.csv.process()
        self.ground_truth_index = 0

    def interpolate_pose(self, i_pose, i_t, f_pose, f_t, x_t):
        o = i_pose + (x_t - i_t) * (f_pose - i_pose) / (f_t - i_t)

        return o

    def calculate_rsme(self, estimated, gt):
        # return np.sum(np.linalg.norm(np.array(self.estimated_pose) - np.array(self.ground_truths), axis=1))
        # np.linalg.norm(np.array(self.estimated_pose) - np.array(self.ground_truths), axis=1)

        rsme = np.mean((estimated - gt) ** 2, axis=0)

        return np.sqrt(rsme)

    def calculate_mae(self, estimated, gt):
        rsme = np.mean(np.abs(estimated - gt), axis=0)

        return np.sqrt(rsme)

    def get_closest_ground_truth(self, t):
        gts = self.csv.sensor_data[DataType.GROUND_TRUTH]

        while self.ground_truth_index >= 0 and t <= gts[self.ground_truth_index].timestamp:
            self.ground_truth_index -= 1

        while self.ground_truth_index < len(gts) - 2 and not (gts[self.ground_truth_index].timestamp <= t and t <= gts[
            self.ground_truth_index + 1].timestamp):
            self.ground_truth_index += 1

        o = self.interpolate_pose(gts[self.ground_truth_index].measurement_data,
                                  gts[self.ground_truth_index].timestamp,
                                  gts[self.ground_truth_index + 1].measurement_data,
                                  gts[self.ground_truth_index + 1].timestamp, t)

        return o

    def process(self):
        output = self.csv.sensor_data[DataType.ODOMETRY]

        estimated = []
        actual = []

        for datapoint in output:
            datapoint: DataPoint

            closest_gt = self.get_closest_ground_truth(datapoint.timestamp)
            estimated.append(datapoint.measurement_data)
            actual.append(closest_gt)

        estimated = np.array(estimated)
        actual = np.array(actual)

        rmse = self.calculate_rsme(estimated, actual)
        rmse_pos = self.calculate_rsme_pose(estimated, actual)
        mae_pos = self.calculate_mae_pose(estimated, actual)
        mae = self.calculate_mae(estimated, actual)

        x = ['x', 'y', 'z', 'v', '$\psi$', '$\dot{\psi}$']
        x2 = ['x', 'y', 'z', 'v', '$\psi$', '$\dot{\psi}$', 'ox', 'oy', 'oz', 'ow']

        un = ['m', 'm', 'm', '$\\frac{m}{s}$', "rad", "rad"]

        for i, r, m, u in zip(x, rmse, mae, un):
            r = np.round(r, 5)
            m = np.round(m, 5)

            print(f"RMSE {i} & {r} {u} & MAE {i} & {m} {u} \\\\")

        print(f"RMSE pose & {rmse_pos:.5f} m & MAE pose & {mae_pos:.5f} m \\\\")

        t = np.arange(0, len(estimated))

        for i in range(estimated.shape[-1]):
            fig: Figure = plt.figure()
            fig.suptitle(x2[i])
            a = fig.gca()
            a.plot(t, estimated[:, i], label='Estimated')
            a.plot(t, actual[:, i], label='Actual')
            a.legend()

        fig = plt.figure()
        a = fig.gca()
        a.set_aspect('equal')
        a.plot(estimated[:, 0], estimated[:, 1], label="Estimated")
        a.plot(actual[:, 0], actual[:, 1], label="Actual")
        a.legend()

        plt.show()

        return rmse

    def calculate_rsme_pose(self, estimated, actual):
        return np.mean(np.linalg.norm((estimated - actual)[:, :3], axis=1) ** 2)

    def calculate_mae_pose(self, estimated, actual):
        return np.mean(np.linalg.norm((estimated - actual)[:, :3], axis=1))


if __name__ == '__main__':
    a = ROSMetrics('data/data.csv')
    a.process()
