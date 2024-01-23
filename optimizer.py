from sklearnex import patch_sklearn

from ukf.datapoint import DataType
from ukf.state import UKFState

patch_sklearn()

from time import time

import numpy as np
from matplotlib import pyplot as plt
from skopt.plots import plot_convergence, plot_objective

from world import UKFRobot, World, ROSWorld, UKFROSRobot

np.random.seed(42)

from skopt import gp_minimize, gbrt_minimize


def run_UKF(x):
    np.random.seed(42)
    err = 1000
    w = World(dt=.1, large_d_p=0.0)

    w.add_uwb((60, 0, 0))
    w.add_uwb((-10, 0, 0))
    w.add_uwb((0, -60, 0))
    w.add_uwb((0, 10, 0))
    # w.add_uwb((1, 1, 0))
    # w.add_uwb((-1, -1, 0))
    # w.add_uwb((3, -5, 0))

    if x[3] ** 2 * (6 + x[5]) + 2 <= 0:
        print("Failure")
        return err

    w.add_robot(UKFRobot(v=1., w=.2, uwb_std=x[0], speed_noise_std=x[1], yaw_rate_noise_std=x[2], alpha=x[3], beta=x[4],
                         k=x[5]))

    for i in range(2000):
        try:
            # for r in w.robots:
            #     if i % 50 == 0:
            #         r['robot'].w = 2 * r['robot'].w
            #     if i % 100 == 0:
            #         r['robot'].w = -r['robot'].w
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
                        r['robot'].v = -r['robot'].v

            # print("Step:", i)
            w.step()
        except np.linalg.LinAlgError:
            print("Failure")
            return err

    rsme = w.calculate_rsme(0)
    # rsme[3 ] /= 100
    # print(rsme)

    data = np.array([5.35021095e-03, 6.35194238e-03, 1.00409941e-40, 1.42628669e-02])

    return np.sum(rsme / data)


def run(world, minimizer, bounds, x0=None, n_calls=60, n=42):
    return minimizer(world, bounds, n_calls=n_calls, random_state=n, x0=x0, n_jobs=6, initial_point_generator='lhs', verbose=True)


def ukf_optimizer():
    start = time()

    bounds = [(0.005, .5), (0.005, 5), (0.005, 2.5), (0.001, 2.), (-1., 3.), (-6., 6.)]
    # x0 = [.1, .1, 2.5, 1, 0, -5]
    x0 = None
    # x0 = [0.0656735202138201, 0.005, 2.1546593167987678, 0.7624456375027308, 1, 0.9248206058828421]
    # x0 = [0.05, 0.05, 1.040286859872973, 3.5881729155829003, 0.042798706624045814, 2.3200423063247566]
    # x0 = [0.09275029262337292, 0.44998668126287167, 1.632607406752226, 0.001, 0, 6]
    x0 = [0.20796177443716282, 2.424418915956577, 0.5860493507682885, 0.7874421557351899, -0.6146625234589184,
          -4.405076722968345]

    # gp_res = run(gp_minimize, bounds, x0, n=None, n_calls=200)
    gp_res = run(run_UKF, gp_minimize, bounds, x0, n=42, n_calls=200)
    # gp_res = run(gbrt_minimize, bounds, x0, n=42)

    # fig:Figure = plt.figure()
    # ax = fig.gca()

    print("Time 1", time() - start)

    start = time()

    plot = plot_convergence(("gp_minimize", gp_res), yscale="log")
    print("Time 2", time() - start)

    plt.plot()

    start = time()
    plot_objective(gp_res, n_points=10, minimum='expected_minimum')

    print("Time 3", time() - start)

    print(gp_res)
    # fig.show()
    plt.show()


def run_UKF_ROS(x):
    err = 4

    if x[3] ** 2 * (6 + x[5]) + 2 <= 0:
        print("Failure")
        return err

    w = ROSWorld('data/out.csv')

    init_pose = w.csv.sensor_data[DataType.GROUND_TRUTH][0]

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
                    )
    )

    try:
        while not w.empty():
            w.step()
    except np.linalg.LinAlgError:
        return err

    rsme = w.calculate_rsme(0)

    return np.sum(rsme)


def create_UKF_ROS(data_file='data/out2.csv'):
    def run_custom_UKF_ROS(x):
        err = 1.5

        if x[3] ** 2 * (6 + x[5]) + 2 <= 0:
            print("Failure")
            return err

        w = ROSWorld(data_file)

        init_pose = w.csv.sensor_data[DataType.GROUND_TRUTH][0]

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
                        imu_std=(x[18],),
                        sensor_used=[DataType.ODOMETRY]
                        )
        )

        try:
            while not w.empty():
                w.step()
        except np.linalg.LinAlgError:
            return err

        rsme = w.calculate_rsme(0)

        return np.sum(rsme)

    return run_custom_UKF_ROS


def ukf_ros_optimizer():
    start = time()

    # uwb_std = x[0],
    # speed_noise_std = x[1],
    # yaw_rate_noise_std = x[2],
    # alpha = x[3],
    # beta = x[4],
    # k = x[5],
    # P = np.diag([x[6], x[7], x[8], x[9], x[10], x[11]])
    bounds = [(0.005, 3), (0.005, 3), (0.005, 3), (0.001, 3.), (-2., 2.), (-6., 0),
              (0.0001, 10), (0.0001, 10), (0.0001, 100), (0.0001, 10), (0.0001, 10), (0.0001, 10),
              (0.01, 20), (0.01, 20), (0.01, 40), (0.01, 5), (0.0001, 3), (0.0001, 3),
              (0.001, 3)]
    x0 = None
    # UWB only
    # 0.47603732478383814
    # x0 = [1.0001, .0101, .01001, 1, 0, -5, 0.0001, 0.0001, 0.0001, .0001, 0.0001, 0.0001]
    # 0.3182534437378754
    # x0 = [0.3005500739580095, 1.7456795243473833, 1.2537087820968702, 0.9756936851785581, 0.12282062791567361, -2.685579681516146, 1.7811686059759613, 8.135101518427398, 0.0001, 0.0001, 9.679152133012991, 6.493196017200628]
    # Odometer + UWB
    # 0.4954515198376885
    x0 = [1.0001, .0101, .01001, 1, 0, -5, 0.0001, 0.0001, 0.0001, .0001, 0.0001, 0.0001]
    # 0.222102102347287
    x0 = [0.5356205657134541, 0.18109875383894544, 0.3323663017573703, 0.001, 2.0, -4.899251280615886,
          2.3463439365430467, 0.0001, 0.0001, 2.8112198335402963, 0.0001, 9.20133215026015, 11.0, 14.0001, 20.9001,
          1.0001, 0.0001, 0.0001]
    # 0.024891601000495363
    x0 = [3.0, 0.688804761738878, 2.4316845626457964, 3.0, -0.9976005928055613, 0.0, 10.0, 7.474000642513913,
          16.74751457478579, 3.986383868394205, 0.9981002081983105, 10.0, 0.01, 0.01, 0.8348157603174587,
          3.1186254198285264, 2.8082493403438566, 1.0723748068270464, .01]
    # UWB + IMU (Yaw only)
    # 2.4686123075984647
    # x0 = [3.0, 0.688804761738878, 2.4316845626457964, 3.0, -0.9976005928055613, 0.0, 10.0, 7.474000642513913,
    #       16.74751457478579, 3.986383868394205, 0.9981002081983105, 10.0, 0.01, 0.01, 0.8348157603174587,
    #       3.1186254198285264, 2.8082493403438566, 1.0723748068270464, .01]
    #

    # gp_res = run(gp_minimize, bounds, x0, n=None, n_calls=200)
    gp_res = run(run_UKF_ROS, gp_minimize, bounds, x0, n=42, n_calls=200)
    # gp_res = run(run_UKF_ROS, gbrt_minimize, bounds, x0, n=42, n_calls=100)

    # fig:Figure = plt.figure()
    # ax = fig.gca()

    print(gp_res)

    print("Time 1", time() - start)

    start = time()

    plot = plot_convergence(("gp_minimize", gp_res), yscale="log")
    print("Time 2", time() - start)

    plt.plot()

    start = time()
    plot_objective(gp_res, n_points=10, minimum='expected_minimum')

    print("Time 3", time() - start)

    # fig.show()
    plt.show()


def ukf_ros_optimizer2():
    start = time()

    # uwb_std = x[0],
    # speed_noise_std = x[1],
    # yaw_rate_noise_std = x[2],
    # alpha = x[3],
    # beta = x[4],
    # k = x[5],
    # P = np.diag([x[6], x[7], x[8], x[9], x[10], x[11]])
    # imu = x[18]
    bounds = [(0.005, 6), (0.005, 6), (0.005, 6), (0.001, 3.), (-2., 2.), (-7., 0),
              (0.0001, 5), (0.0001, 5), (0.0001, 5), (0.0001, 5), (0.0001, 5), (0.0001, 5),
              (0.01, 5), (0.01, 5), (0.01, 5), (0.01, 5), (0.0001, 5), (0.0001,5 ),
              (0.001, 5)]

    # 4.863677449166163
    x0 = [1.0001, 3.9001, 4.9001, 1, 0, -5, 0.0001, 0.0001, 0.0001,
          2.0001, 0.0001, 0.0001, 11.0, 14.0001, 20.9001, 1.0001, 0.0001, 0.0001, 1]
    # 1.0233502515039579
    x0 = [1., 1., 1., 1., 0., -5, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    #
    x0 = [2.659499784397555, 1.0831178839141666, 2.1359515250278855, 0.8919490628763381, -1.8636939535841257, -5.016355891607367, 3.493890077753967, 2.8665647051912844, 0.0001, 1.36670195424429, 1.1333357613675807, 0.0001, 0.34671604921321414, 1.640808677965142, 2.0324656526567084, 5.0, 0.6639031999283294, 5.0, 0.9609048716515619]
    # x0 = [2.825966362705224, 1.9316060804766901, 0.03264254802240296, 2.698241689740617, 1.9911215771524704,
    #       -4.771320278302439, 3.197382259643369, 10.0, 9.141328450229278, 1.83098378534384, 10.0, 10.0,
    #       0.34521128571916754, 16.05322885480677, 0.06593248652230164, 1.2495450013865828, 0.3282816961808588,
    #       2.9894966141249806, 1.4103946098233913]

    # gp_res = run(gp_minimize, bounds, x0, n=None, n_calls=200)
    gp_res = run(create_UKF_ROS('data/out2.csv'), gp_minimize, bounds, x0, n=42, n_calls=200)
    # gp_res = run(create_UKF_ROS('data/out2.csv'), gbrt_minimize, bounds, x0, n=42, n_calls=60)
    # gp_res = run(run_UKF_ROS, gbrt_minimize, bounds, x0, n=42, n_calls=100)

    print(gp_res)

    print("Time 1", time() - start)

    start = time()

    plot = plot_convergence(("gp_minimize", gp_res), yscale="log")
    print("Time 2", time() - start)

    plt.plot()

    start = time()
    plot_objective(gp_res, n_points=10, minimum='expected_minimum')

    print("Time 3", time() - start)

    plt.show()


if __name__ == '__main__':
    # ukf_optimizer()
    # ukf_ros_optimizer()
    ukf_ros_optimizer2()
# conda install scikit-learn-intelex
# python -m sklearnex my_application.py
