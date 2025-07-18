import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import mean_absolute_error, mean_squared_error

# === Класс DroneNode ===
class DroneNode:
    def __init__(self, drone_id, x_true, gnss_noise_std=1.0, ins_noise_std=0.5):
        self.id = drone_id
        self.x_true = np.array(x_true)
        # GNSS измерение с шумом
        self.z_gnss = self.x_true + np.random.normal(0, gnss_noise_std, size=2)
        # INS оценка на основе предыдущей позиции + шум
        self.prev_position = self.x_true - np.random.normal(0, 1.0, size=2)
        delta_u = self.x_true - self.prev_position
        self.x_ins = self.prev_position + delta_u + np.random.normal(0, ins_noise_std, size=2)
        self.range_measurements = {}

    def measure_range_to(self, other, range_noise_std=0.2):
        dist = np.linalg.norm(self.x_true - other.x_true)
        self.range_measurements[other.id] = dist + np.random.normal(0, range_noise_std)


# === Класс LeaderNode (SwarmRaft) ===
class LeaderNode:
    def __init__(self, drones, range_noise_std=0.2, gnss_var=1.0):
        self.drones = {d.id: d for d in drones}
        self.range_noise_std = range_noise_std
        self.gnss_var = gnss_var
        # Порог T = 3 * sqrt(var_GNSS + var_range)
        self.T = 3 * np.sqrt(gnss_var + range_noise_std**2)

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
            if sum(vlist) <= -(n - f):
                faulty.add(i)
        return faulty

    def recover_positions(self, faulty):
        recovered = {}
        for i in faulty:
            estimates = []
            for j in self.drones:
                if i == j: continue
                estimates.append(self.fuse_estimate(self.drones[j], self.drones[i]))
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


# === Настройка и запуск симуляции ===
np.random.seed(42)
N = 5
# Генерация истинных позиций
positions = np.random.rand(N, 2) * 20
drones = [DroneNode(i, positions[i]) for i in range(N)]

# Измерения междроновых расстояний
for i in range(N):
    for j in range(N):
        if i != j:
            drones[i].measure_range_to(drones[j])

# Первый шаг консенсуса (baseline)
leader = LeaderNode(drones)
baseline_positions, baseline_faulty = leader.step_consensus()

# Спуфинг GNSS для дрона D2
drones[2].z_gnss += np.array([15.0, -15.0])

# Повтор консенсуса после атаки
leader_attacked = LeaderNode(drones)
leader_attacked.T = 2 * np.sqrt(leader_attacked.gnss_var + leader_attacked.range_noise_std**2)
final_attacked, faulty_attacked = leader_attacked.step_consensus()

# === Вычисление метрик ===
true_pos      = np.array([d.x_true for d in drones])
gnss_pos      = np.array([d.z_gnss for d in drones])
recovered_pos = np.array([final_attacked[i] for i in range(N)])

mae_gnss = mean_absolute_error(true_pos, gnss_pos)
mae_rec  = mean_absolute_error(true_pos, recovered_pos)
mse_gnss = mean_squared_error(true_pos, gnss_pos)
mse_rec  = mean_squared_error(true_pos, recovered_pos)
rmse_gnss = np.sqrt(mse_gnss)
rmse_rec  = np.sqrt(mse_rec)

# === Визуализация: два subplot в одной фигуре ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Subplot 1: позиции и GNSS / INS / Recovered ---
ax1.set_title("SwarmRaft: Positions & GNSS Spoofing")
for d in drones:
    ax1.scatter(*d.x_true, color='green', s=100, zorder=1, label='True' if d.id==0 else "")
    ax1.scatter(*d.x_ins,  color='red',   marker='^', s=100, zorder=2, label='INS'  if d.id==0 else "")
    ax1.scatter(*final_attacked[d.id], color='black', marker='*', s=120, zorder=3,
                label='Recovered' if d.id==0 else "")
    ax1.plot( d.z_gnss[0], d.z_gnss[1],
              marker='x', markersize=14, markeredgewidth=3,
              color='blue', zorder=4, label='GNSS' if d.id==0 else "" )
    ax1.text(d.x_true[0]+0.3, d.x_true[1]+0.3, f"D{d.id}", fontsize=9, zorder=5)
for fid in faulty_attacked:
    fp = final_attacked[fid]
    ax1.text(fp[0], fp[1]-1, "FAULT", color='crimson', fontsize=10,
             ha='center', zorder=5)
ax1.set_xlim(0,22); ax1.set_ylim(0,22)
ax1.grid(True); ax1.set_aspect('equal', 'box')
ax1.legend(loc='upper right')

# --- Subplot 2: bar-chart ошибок ---
ax2.set_title("Error Comparison")
metrics = ['MAE', 'RMSE']
vals_gnss = [mae_gnss, rmse_gnss]
vals_rec  = [mae_rec,  rmse_rec]
x = np.arange(len(metrics))
width = 0.35
ax2.bar(x - width/2, vals_gnss, width=width, label='GNSS')
ax2.bar(x + width/2, vals_rec,  width=width, label='Recovered')
ax2.set_xticks(x); ax2.set_xticklabels(metrics)
ax2.set_ylabel("Error (m)")
ax2.grid(axis='y')
ax2.legend()

plt.tight_layout()
plt.show()

# === Печать метрик в консоль ===
print("\n=== SwarmRaft Attack Recovery Metrics ===")
print(f"MAE (GNSS):       {mae_gnss:.3f} m")
print(f"MAE (Recovered):  {mae_rec:.3f} m")
print(f"RMSE (GNSS):      {rmse_gnss:.3f} m")
print(f"RMSE (Recovered): {rmse_rec:.3f} m")
