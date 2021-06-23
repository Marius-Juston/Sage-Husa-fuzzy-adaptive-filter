class Localization:
    def __init__(self):
        self.odometry = []

    def trilaterate_position(self, range_data, initial_pose=(0, 0, 0)):
        if len(self.odometry_data) > len(range_data):
            odometry = self.find_closest_odometry(range_data)
        else:
            odometry = np.zeros((len(range_data), 4))

        # import json

        # np.save("/home/marius/catkin_ws/src/uwb_localization/odometry", odometry)

        # print(odometry)
        # print(self.odom_times)
        # print(self.odometry_data)

        # new_d = range_data

        # for i in range(len(new_d)):
        #     new_d[i]['pose'] = new_d[i]['pose'].tolist()

        # json.dump( new_d, open("/home/marius/catkin_ws/src/uwb_localization/range_data.json", 'w'), )
        # print("range_data.json")

        res = least_squares(self.trilateration_function, initial_pose, args=(range_data, odometry))

        if res.cost > 50:
            local_mininum, error = self.check_for_local_mininum(res, range_data, odometry)
        else:
            local_mininum = res.x
            error = self.rmse(res.fun)

        return np.array([local_mininum[0], local_mininum[1], 0, 0, local_mininum[2], 0]), error

    def rmse(self, residuals):
        return np.sqrt(np.mean((residuals) ** 2))

    def trilateration_function(self, input_x, distances, odometry_data):
        # x[0] = x_start
        # x[1] = y_start
        # x[2] = theta

        _, _, theta = input_x

        xy = input_x[:2]

        residuals = [
            # (x1 - x2) ** 2 + (y1 - y2) ** 2 - self.d ** 2,
        ]

        for i, distance in enumerate(distances):
            anchor = distance['pose']
            tagID = distance['tagID']
            distance = distance['range']

            odometry = odometry_data[i]
            # 0 = x, 1 = y, 2 = z, 3 = theta
            xy_odom = odometry[:2]
            theta_odom = odometry[3]

            z = self.loc.tag_offset[tagID][2] + odometry[2]

            xy_tag = self.loc.tag_offset[tagID][:2]

            xy_world = self.rotate(xy_odom, theta) + xy
            xy_tag = self.rotate(xy_tag, theta + theta_odom) + xy_world

            x, y = xy_tag

            residuals.append((x - anchor[0]) ** 2 + (y - anchor[1]) ** 2 + (z - anchor[2]) ** 2 - distance ** 2)

        return residuals

    def rotate(self, xy, rot_angle):
        c = np.cos(rot_angle)
        s = np.sin(rot_angle)

        result = np.matmul(np.array([[c, -s],
                                     [s, c]]), xy)

        return result
