import numpy as np


class DroneNode:
    def __init__(self, drone_id, x_true, gnss_noise_std=1.0, ins_noise_std=0.5):
        """
        Инициализация дрона:
        :param drone_id: уникальный идентификатор дрона
        :param x_true: истинная позиция дрона (np.array, размерность 2 или 3)
        :param gnss_noise_std: стандартное отклонение шума GNSS
        :param ins_noise_std: стандартное отклонение шума INS
        """
        self.id = drone_id
        self.x_true = np.array(x_true)  # Ground truth position
        self.z_gnss = self.x_true + np.random.normal(0, gnss_noise_std, size=self.x_true.shape)  # GNSS measurement

        # Симуляция "предыдущего шага" и движения дрона
        self.prev_position = self.x_true - np.random.normal(0, 1.0, size=self.x_true.shape)
        delta_u = self.x_true - self.prev_position
        self.x_ins = self.prev_position + delta_u + np.random.normal(0, ins_noise_std, size=self.x_true.shape)  # INS estimate

        self.range_measurements = {}  # {other_id: measured_distance}
        self.fused_estimates = {}     # {other_id: fused estimate of other drone's position}

    def measure_range_to(self, other_drone, range_noise_std=0.2):
        """
        Измеряет расстояние до другого дрона с шумом
        :param other_drone: объект DroneNode
        :param range_noise_std: стандартное отклонение шума измерения расстояния
        """
        true_distance = np.linalg.norm(self.x_true - other_drone.x_true)
        noisy_distance = true_distance + np.random.normal(0, range_noise_std)
        self.range_measurements[other_drone.id] = noisy_distance

    def __repr__(self):
        return f"<DroneNode id={self.id}, true={self.x_true}, gnss={self.z_gnss}, ins={self.x_ins}>"
