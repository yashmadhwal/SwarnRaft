import pandas as pd
import matplotlib.pyplot as plt

# Загружаем динамически сгенерированные данные
df = pd.read_csv("swarmraft_multi_attacks_dynamic.csv")

# Группируем и считаем среднее + стандартное отклонение
grouped = df.groupby("Num_Attacked").agg(
    MAE_GNSS_mean=("MAE_GNSS", "mean"),
    MAE_GNSS_std=("MAE_GNSS", "std"),
    MAE_Rec_mean=("MAE_Recovered", "mean"),
    MAE_Rec_std=("MAE_Recovered", "std"),
    RMSE_GNSS_mean=("RMSE_GNSS", "mean"),
    RMSE_GNSS_std=("RMSE_GNSS", "std"),
    RMSE_Rec_mean=("RMSE_Recovered", "mean"),
    RMSE_Rec_std=("RMSE_Recovered", "std")
).reset_index()

x = grouped["Num_Attacked"]

# Рисуем
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.errorbar(x, grouped["MAE_GNSS_mean"], yerr=grouped["MAE_GNSS_std"], fmt='-o', capsize=5, label="GNSS MAE")
ax1.errorbar(x, grouped["MAE_Rec_mean"], yerr=grouped["MAE_Rec_std"], fmt='-o', capsize=5, label="Recovered MAE")
ax1.set_title("MAE vs Number of Attacked Drones")
ax1.set_xlabel("Number of Attacked Drones")
ax1.set_ylabel("Mean Absolute Error (m)")
ax1.grid(True)
ax1.legend()

ax2.errorbar(x, grouped["RMSE_GNSS_mean"], yerr=grouped["RMSE_GNSS_std"], fmt='-o', capsize=5, label="GNSS RMSE")
ax2.errorbar(x, grouped["RMSE_Rec_mean"], yerr=grouped["RMSE_Rec_std"], fmt='-o', capsize=5, label="Recovered RMSE")
ax2.set_title("RMSE vs Number of Attacked Drones")
ax2.set_xlabel("Number of Attacked Drones")
ax2.set_ylabel("Root Mean Square Error (m)")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()
