import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Симулируем реализацию DroneNode и LeaderNode в одном скрипте
# Чтобы показать работу полной симуляции

class DroneNode:
    def __init__(self, drone_id, x_true, gnss_noise_std=1.0, ins_noise_std=0.5):
        self.id = drone_id
        self.x_true = np.array(x_true)
        self.z_gnss = self.x_true + np.random.normal(0, gnss_noise_std, size=2)
        self.prev_position = self.x_true - np.random.normal(0, 1.0, size=2)
        delta_u = self.x_true - self.prev_position
        self.x_ins = self.prev_position + delta_u + np.random.normal(0, ins_noise_std, size=2)
        self.range_measurements = {}

    def measure_range_to(self, other, range_noise_std=0.2):
        dist = np.linalg.norm(self.x_true - other.x_true)
        self.range_measurements[other.id] = dist + np.random.normal(0, range_noise_std)


class LeaderNode:
    def __init__(self, drones, range_noise_std=0.2, gnss_var=1.0, ins_var=0.25):
        self.drones = {d.id: d for d in drones}
        self.range_noise_std = range_noise_std
        self.gnss_var = gnss_var
        self.T = 3 * np.sqrt(gnss_var + range_noise_std ** 2)

    def fuse_estimate(self, from_d, to_d):
        dij = from_d.range_measurements[to_d.id]
        xj_ins = from_d.x_ins
        xi_prev = to_d.prev_position
        direction = xi_prev - xj_ins
        norm = np.linalg.norm(direction)
        if norm == 0:
            return to_d.z_gnss
        x_range = xj_ins + (dij / norm) * direction
        var_range = self.range_noise_std ** 2
        alpha = var_range / (self.gnss_var + var_range)
        return alpha * to_d.z_gnss + (1 - alpha) * x_range

    def compute_votes(self):
        votes = defaultdict(list)
        for i in self.drones:
            for j in self.drones:
                if i == j: continue
                fused = self.fuse_estimate(self.drones[j], self.drones[i])
                residual = np.linalg.norm(fused - self.drones[i].z_gnss)
                votes[i].append(1 if residual <= self.T else -1)
        return votes

    def detect_faulty_nodes(self, votes, f=1):
        faulty = set()
        n = len(self.drones)
        for i, vlist in votes.items():
            total = sum(vlist)
            if total <= -(n - f):
                faulty.add(i)
        return faulty

    def recover_positions(self, faulty):
        recovered = {}
        for i in faulty:
            estimates = []
            for j in self.drones:
                if i == j: continue
                fused = self.fuse_estimate(self.drones[j], self.drones[i])
                estimates.append(fused)
            recovered[i] = np.median(estimates, axis=0)
        return recovered

    def step_consensus(self, f=1):
        votes = self.compute_votes()
        faulty = self.detect_faulty_nodes(votes, f)
        recovered = self.recover_positions(faulty)
        final = {}
        for i in self.drones:
            final[i] = recovered[i] if i in recovered else self.drones[i].z_gnss
        return final, faulty


# === Симуляция ===
np.random.seed(42)
N = 5
positions = np.random.rand(N, 2) * 20
drones = [DroneNode(i, positions[i]) for i in range(N)]
for i in range(N):
    for j in range(N):
        if i != j:
            drones[i].measure_range_to(drones[j])

leader = LeaderNode(drones)
leader.T = 2 * np.sqrt(leader.gnss_var + leader.range_noise_std**2)
final_positions, faulty_nodes = leader.step_consensus()

# === Визуализация ===
plt.figure(figsize=(8, 8))

# 1) Истинные позиции ◯
for d in drones:
    plt.scatter(
        *d.x_true,
        c='green',
        s=100,
        label='True' if d.id == 0 else "",
        zorder=1
    )

# 2) INS-оценки △
for d in drones:
    plt.scatter(
        *d.x_ins,
        c='red',
        marker='^',
        s=100,
        label='INS' if d.id == 0 else "",
        zorder=2
    )

# 3) Восстановленные позиции ★
for idx, pos in final_positions.items():
    plt.scatter(
        *pos,
        c='black',
        marker='*',
        s=120,
        label='Recovered' if idx == 0 else "",
        zorder=3
    )

# 4) GNSS-измерения ✕ (через plot, крупные крестики)
for d in drones:
    plt.plot(
        d.z_gnss[0],
        d.z_gnss[1],
        marker='x',
        markersize=14,
        markeredgewidth=3,
        color='blue',
        label='GNSS' if d.id == 0 else "",
        zorder=4
    )

# 5) Подписи и метки FAULT
for d in drones:
    plt.text(
        d.x_true[0] + 0.3,
        d.x_true[1] + 0.3,
        f'D{d.id}',
        fontsize=9,
        zorder=5
    )
for idx in faulty_nodes:
    fp = final_positions[idx]
    plt.text(
        fp[0],
        fp[1] - 1.0,
        'FAULT',
        color='crimson',
        fontsize=10,
        ha='center',
        zorder=5
    )

plt.title("SwarmRaft Consensus & Recovery")
plt.xlabel("X Coordinate (m)")
plt.ylabel("Y Coordinate (m)")
plt.legend(loc='upper right')
plt.grid(True)
plt.axis('equal')
plt.xlim(0, 22)
plt.ylim(0, 22)
plt.show()
