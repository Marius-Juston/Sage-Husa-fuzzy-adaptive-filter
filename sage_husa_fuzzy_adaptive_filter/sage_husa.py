from sage_husa_fuzzy_adaptive_filter.measurement_predictor import MeasurementPredictor
from sage_husa_fuzzy_adaptive_filter.state_predictor import StatePredictor
from sage_husa_fuzzy_adaptive_filter.state_updater import StateUpdater


class SageHusaAdaptiveFilter:
    def __init__(self) -> None:
        super().__init__()

        self.state_predictor = StatePredictor()
        self.measurement_predictor = MeasurementPredictor()
        self.state_updater = StateUpdater()

        # state
        # [px, py, pz, vx, vy, vz, orienx, orieny, orienz, orienw, ax, ay, ax

        self.x = None
        self.P = None

    def process(self, data):
        dt = (data.timestamp - self.timestamp) / 1e9  # ns to seconds
        # dt = (data.timestamp - self.timestamp)  # seconds

        if dt < 0.0001:
            dt = 0.0001

        # STATE PREDICTION
        # get predicted state and covariance of predicted state, predicted sigma points in state space
        self.state_predictor.process(self.x, self.P, dt)
        self.x = self.state_predictor.x
        self.P = self.state_predictor.P
        sigma_x = self.state_predictor.sigma

        # MEASUREMENT PREDICTION
        # get predicted measurement, covariance of predicted measurement, predicted sigma points in measurement space
        self.measurement_predictor.process(sigma_x, data)
        predicted_z = self.measurement_predictor.z
        S = self.measurement_predictor.S
        sigma_z = self.measurement_predictor.sigma_z

        # STATE UPDATE
        # updated the state and covariance of state... also get the nis
        self.state_updater.process(self.x, predicted_z, data.measurement_data, S, self.P, sigma_x, sigma_z,
                                   data.data_type)
        self.x = self.state_updater.x
        self.P = self.state_updater.P

        self.timestamp = data.timestamp
