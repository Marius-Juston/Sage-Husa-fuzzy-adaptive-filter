from sklearnex import patch_sklearn

patch_sklearn()

from time import time

import numpy as np
from matplotlib import pyplot as plt
from skopt.plots import plot_convergence, plot_objective

from world import UKFRobot, World

np.random.seed(42)

from skopt import gp_minimize


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
    return minimizer(world, bounds, n_calls=n_calls, random_state=n, x0=x0, n_jobs=6, initial_point_generator='lhs')


if __name__ == '__main__':
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

# conda install scikit-learn-intelex
# python -m sklearnex my_application.py
