import numpy as np
import matplotlib.pyplot as plt
from drone_node import DroneNode
from leader_node import LeaderNode

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

plt.figure(figsize=(8, 8))
for d in drones:
    plt.scatter(*d.x_true, c='green', s=100, label='True' if d.id == 0 else "")
    plt.scatter(*d.z_gnss, c='blue', marker='x', label='GNSS' if d.id == 0 else "")
    plt.scatter(*d.x_ins, c='red', marker='^', label='INS' if d.id == 0 else "")
    plt.text(d.x_true[0] + 0.3, d.x_true[1] + 0.3, f'D{d.id}', fontsize=9)

for i, pos in final_positions.items():
    plt.scatter(*pos, c='black', marker='*', s=120, label='Recovered' if i == 0 else "")
for i in faulty_nodes:
    plt.text(final_positions[i][0], final_positions[i][1] - 1.0, 'FAULT', color='crimson', fontsize=10)

plt.title("SwarmRaft: Baseline Recovery")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.xlim(0, 22)
plt.ylim(0, 22)
plt.show()
