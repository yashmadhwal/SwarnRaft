
# SwarmRaft: Resilient Consensus-Based Localization for UAV Swarms

SwarmRaft is a modular simulation framework designed to evaluate robust localization in UAV swarms under adversarial conditions such as GNSS spoofing and sensor faults. The system integrates Raft-inspired consensus, GNSS/INS fusion, peer-to-peer ranging, and voting-based recovery mechanisms to maintain coordination and safety.

## 🧠 Core Features

- **Crash-tolerant Raft-style consensus**
- **GNSS + INS fusion for state estimation**
- **Peer-based distance measurements for fault detection**
- **Byzantine-resilient majority voting**
- **Scalable simulation across swarm sizes and attack scenarios**

## 🗂️ Repository Structure

| File                         | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| `drone_node.py`             | Defines `DroneNode`: GNSS/INS model, inter-node ranging                    |
| `leader_node.py`            | Implements consensus leader: fusion, voting, and recovery logic            |
| `attack_simulation.py`      | Simulates swarm behavior under GNSS spoofing and distance perturbations     |
| `generate_attack_experiments.py` | Automates Monte Carlo trials for statistical robustness evaluation   |
| `run_experiment.py`         | Launches simulation with configurable swarm and attack parameters          |
| `plot_dynamic_results.py`   | Visualizes MAE/RMSE trends and boxplots for comparative analysis           |
| `simulator.py`              | Optional visual demonstration of swarm recovery behavior                   |

## 📊 Evaluation Metrics

- **MAE** – Mean Absolute Error
- **RMSE** – Root Mean Squared Error  
Both metrics are computed for GNSS-only vs. SwarmRaft-recovered positions.

## 🚀 Getting Started

```bash
pip install -r requirements.txt

# Run a sample simulation
python run_experiment.py --n_drones 10 --n_attackers 3

# Plot results
python plot_dynamic_results.py
```

## 📄 Citation

If you use this framework in academic work, please cite the associated paper (coming soon).

## 📬 Contact

For questions, contributions, or collaborations, please contact the Authors of the scientific paper (coming soon).
