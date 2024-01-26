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
        o = i_pose + (x_t - i_t) * (f_pose - i_pose) / (f_t - i_t + 1e-6)

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

    def process(self, confidence=None):
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

        j = 0

        confidence = np.round(confidence, 3)

        for i, r, m, u in zip(x, rmse, mae, un):
            r = np.round(r, 4)
            m = np.round(m, 4)

            if confidence is None:
                print(f"RMSE {i} & {r} {u} & MAE {i} & {m} {u} \\\\")
            else:
                print(f"RMSE {i} & {r} {u} $\pm$ {confidence[j]} {u} & MAE {i} & {m} {u} $\pm$ {confidence[j + len(rmse) + 1]} {u} \\\\")

            j += 1

        print(f"RMSE pose & {rmse_pos:.5f} m $\pm$ {confidence[len(rmse)]} m & MAE pose & {mae_pos:.5f} m $\pm$ {confidence[-1]} m \\\\")

        t = np.arange(0, len(estimated))

        for i in range(estimated.shape[-1]):
            fig: Figure = plt.figure()
            fig.suptitle(x2[i])
            a = fig.gca()
            a.plot(t, estimated[:, i], label='Estimated')
            a.plot(t, actual[:, i], label='Actual')

            a.set_xlabel('Time step (50 hz)')
            a.set_ylabel(f"{x2[i]}-coordinate (m)")
            a.legend()

        fig = plt.figure()
        a = fig.gca()
        a.set_aspect('equal')
        a.plot(estimated[:, 0], estimated[:, 1], label="Estimated")
        a.plot(actual[:, 0], actual[:, 1], label="Actual")
        a.legend()

        plt.show()

        return rmse

    def compute_errors(self):
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

        return np.stack([*rmse, rmse_pos, *mae, mae_pos])

    def calculate_rsme_pose(self, estimated, actual):
        return np.mean(np.linalg.norm((estimated - actual)[:, :3], axis=1) ** 2)

    def calculate_mae_pose(self, estimated, actual):
        return np.mean(np.linalg.norm((estimated - actual)[:, :3], axis=1))


def compute_interval(folder='temp'):
    data = []

    for i in range(5):
        a = ROSMetrics(f'{folder}/out{i + 1}.csv')
        data.append(a.compute_errors())

    data = np.array(data)

    stds = np.std(data, axis=0)

    z_star = 1.96

    confidence = z_star * stds / np.sqrt(data.shape[0])

    print(np.round(confidence, 5))

    return confidence


if __name__ == '__main__':
    confidence = compute_interval()

    a = ROSMetrics('data/data.csv')
    # a = ROSMetrics('temp/out1.csv')
    a.process(confidence)
