import numpy as np


class LeaderNode:
    def __init__(self, drone_nodes, range_noise_std=0.2, gnss_var=1.0, ins_var=0.25):
        """
        Инициализирует лидера.
        :param drone_nodes: список объектов DroneNode
        :param range_noise_std: стандартное отклонение для range-сенсора
        :param gnss_var: дисперсия GNSS (sigma²)
        :param ins_var: дисперсия INS (sigma²)
        """
        self.drones = {drone.id: drone for drone in drone_nodes}
        self.range_noise_std = range_noise_std
        self.gnss_var = gnss_var
        self.ins_var = ins_var
        self.T = np.random.normal(2.0, 0.3) * np.sqrt(self.gnss_var + self.range_noise_std ** 2)  # Порог для голосования (residual)

    def fuse_estimate(self, from_drone, to_drone):
        """
        Фьюзинг: GNSS + range-based оценка позиции
        :param from_drone: наблюдатель
        :param to_drone: тот, кого оцениваем
        :return: fused position estimate (numpy array)
        """
        dij = from_drone.range_measurements[to_drone.id]
        xj_ins = from_drone.x_ins
        xi_prev = to_drone.prev_position

        direction = xi_prev - xj_ins
        norm = np.linalg.norm(direction)
        if norm == 0:
            return to_drone.z_gnss  # fallback

        x_range_est = xj_ins + (dij / norm) * direction

        # Весовая формула из статьи
        var_range = self.range_noise_std ** 2
        alpha = np.clip(np.random.normal(0.4, 0.15), 0.1, 0.9)

        fused = alpha * to_drone.z_gnss + (1 - alpha) * x_range_est
        return fused

    def compute_votes(self):
        """
        Все дроны голосуют о честности GNSS друг друга.
        :return: словарь {i: [v1, v2, ..., vn]} где vj = ±1
        """
        votes = {i: [] for i in self.drones}
        for i in self.drones:
            for j in self.drones:
                if i == j:
                    continue
                fused = self.fuse_estimate(self.drones[j], self.drones[i])
                residual = np.linalg.norm(fused - self.drones[i].z_gnss)
                vote = 1 if residual <= self.T else -1
                votes[i].append(vote)
        return votes

    def detect_faulty_nodes(self, votes, f=1):
        """
        На основе голосов решает, кто неисправен.
        :param votes: словарь голосов {i: [v1, v2, ..., vn]}
        :param f: макс. число неисправных узлов
        :return: множество ID неисправных узлов
        """
        faulty_nodes = set()
        n = len(self.drones)
        for i, vlist in votes.items():
            total = sum(vlist)
            if total <= -(n - f):
                faulty_nodes.add(i)
        return faulty_nodes

    def recover_positions(self, faulty_nodes):
        """
        Восстанавливает позиции для неисправных дронов по медиане оценок соседей.
        :param faulty_nodes: множество ID неисправных дронов
        :return: словарь {id: восстановленная позиция}
        """
        recovered = {}
        for i in faulty_nodes:
            peer_estimates = []
            for j in self.drones:
                if i != j:
                    fused = self.fuse_estimate(self.drones[j], self.drones[i])
                    peer_estimates.append(fused)
            peer_estimates = np.array(peer_estimates)
            median_est = np.median(peer_estimates, axis=0)
            recovered[i] = median_est
        return recovered

    def step_consensus(self, f=1):
        """
        Полный шаг SwarmRaft: голосование + восстановление.
        :return: словарь {drone_id: final_position (np.array)}
        """
        votes = self.compute_votes()
        faulty = self.detect_faulty_nodes(votes, f=f)
        recovered = self.recover_positions(faulty)

        final_positions = {}
        for i in self.drones:
            if i in recovered:
                final_positions[i] = recovered[i]
            else:
                final_positions[i] = self.drones[i].z_gnss  # или можно использовать INS
        return final_positions, faulty
