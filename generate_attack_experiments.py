import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# === Общие параметры ===
np.random.seed(42)
repeats = 10
bias = 15.0
N_values = [5, 10, 15]  # можно менять
records = []

# === Симуляция с перебором N и f (число атакованных) ===
for N in N_values:
    for f in range(1, N):  # атакуем от 1 до N-1 дронов
        for _ in range(repeats):
            positions = np.random.rand(N, 2) * 20

            class Drone:
                def __init__(self, drone_id, pos):
                    self.id = drone_id
                    self.x_true = pos
                    self.z_gnss = self.x_true + np.random.normal(0, 1.0, 2)
                    self.prev_position = self.x_true - np.random.normal(0, 1.0, 2)
                    delta = self.x_true - self.prev_position
                    ins_std = np.random.uniform(0.3, 1.0)
                    self.x_ins = self.prev_position + delta + np.random.normal(0, ins_std, 2)
                    self.range_measurements = {}

            drones = [Drone(i, positions[i]) for i in range(N)]
            for i in range(N):
                for j in range(N):
                    if i != j:
                        dist = np.linalg.norm(drones[i].x_true - drones[j].x_true)
                        drones[i].range_measurements[j] = dist + np.random.normal(0, 0.2)

            # атака
            attacked_ids = np.random.choice(range(N), size=f, replace=False)
            for idx in attacked_ids:
                drones[idx].z_gnss += np.array([bias, -bias])
                for j in range(N):
                    if j != idx:
                        drones[idx].range_measurements[j] += np.random.uniform(-2, 2)

            class Leader:
                def __init__(self, drones):
                    self.drones = {d.id: d for d in drones}
                    self.range_noise_std = 0.2
                    self.gnss_var = 1.0
                    self.T = np.random.normal(2.0, 0.3) * np.sqrt(self.gnss_var + self.range_noise_std ** 2)

                def fuse(self, from_d, to_d):
                    dij = from_d.range_measurements[to_d.id]
                    xj_ins = from_d.x_ins
                    xi_prev = to_d.prev_position
                    direction = xi_prev - xj_ins
                    norm = np.linalg.norm(direction)
                    if norm == 0:
                        return to_d.z_gnss
                    x_range = xj_ins + (dij / norm) * direction
                    alpha = np.clip(np.random.normal(0.4, 0.15), 0.1, 0.9)
                    return alpha * to_d.z_gnss + (1 - alpha) * x_range

                def detect_faulty(self, votes, f=1):
                    faulty = set()
                    n = len(self.drones)
                    for i, vlist in votes.items():
                        if sum(vlist) <= -(n - f):
                            faulty.add(i)
                    return faulty

                def step(self, f=1):
                    votes = {i: [] for i in self.drones}
                    for i in self.drones:
                        for j in self.drones:
                            if i == j:
                                continue
                            fused = self.fuse(self.drones[j], self.drones[i])
                            residual = np.linalg.norm(fused - self.drones[i].z_gnss)
                            attacker = j in attacked_ids
                            vote = 1 if residual <= self.T else -1
                            vote = vote if not attacker else np.random.choice([1, -1])
                            votes[i].append(vote)
                    faulty = self.detect_faulty(votes, f)
                    recovered = {}
                    for i in faulty:
                        estimates = []
                        for j in self.drones:
                            if i != j:
                                estimates.append(self.fuse(self.drones[j], self.drones[i]))
                        recovered[i] = np.median(estimates, axis=0)
                    final = {}
                    for i in self.drones:
                        final[i] = recovered[i] if i in recovered else self.drones[i].z_gnss
                    return final, faulty

            leader = Leader(drones)
            final_pos, faulty = leader.step(f=f)

            true_pos = np.array([d.x_true for d in drones])
            gnss_pos = np.array([d.z_gnss for d in drones])
            recovered_pos = np.array([final_pos[i] for i in range(N)])

            mae_gnss = mean_absolute_error(true_pos, gnss_pos)
            mae_rec = mean_absolute_error(true_pos, recovered_pos)
            rmse_gnss = np.sqrt(mean_squared_error(true_pos, gnss_pos))
            rmse_rec = np.sqrt(mean_squared_error(true_pos, recovered_pos))

            records.append({
                "Num_Drones": N,
                "Num_Attacked": f,
                "MAE_GNSS": mae_gnss,
                "MAE_Recovered": mae_rec,
                "RMSE_GNSS": rmse_gnss,
                "RMSE_Recovered": rmse_rec
            })

# === Сохраняем в CSV
# === Сохраняем в единый CSV для всех N
df = pd.DataFrame(records)
csv_path = "swarmraft_scaling_experiment.csv"
df.to_csv(csv_path, index=False)

print(f"\n✅ CSV saved as: {csv_path}")
print(df.head())

