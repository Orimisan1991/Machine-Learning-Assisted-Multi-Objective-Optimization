# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# =========================================================
# 2. GLOBAL VARIABLES
# =========================================================
temps = [20,30,40,50,60,70]
concs = [0.3,0.2,0.1,0.05,0.025,0.0125]

# =========================================================
# 3. ALL DATA TABLES
# =========================================================

# --- Thermal Conductivity ---
k_nonmag = [
[0.63841,0.62616,0.62186,0.61327,0.60467,0.59823,0.58498,0.572],
[0.65087,0.63905,0.63368,0.62616,0.62186,0.61112,0.60542,0.5836],
[0.66311,0.6528,0.64678,0.63905,0.63475,0.62616,0.61864,0.592],
[0.67665,0.66483,0.66161,0.65194,0.64764,0.64335,0.63153,0.603],
[0.68954,0.67772,0.67235,0.66483,0.66053,0.65362,0.64442,0.6134],
[0.70351,0.69061,0.68301,0.67772,0.67343,0.66913,0.65731,0.621],
[0.71532,0.70651,0.6989,0.69061,0.68632,0.67852,0.67128,0.63066],
[0.72821,0.71832,0.71317,0.70458,0.69921,0.68992,0.68309,0.639],
[0.7411,0.73036,0.72606,0.7164,0.70995,0.70012,0.69169,0.649]
]

k_mag = [
[0.63798,0.62938,0.62616,0.62079,0.61434,0.6079,0.60252,0.572],
[0.65087,0.64227,0.6369,0.6326,0.62616,0.62304,0.61864,0.5836],
[0.66762,0.65624,0.6513,0.6455,0.64231,0.63811,0.63013,0.592],
[0.67665,0.66913,0.66376,0.65731,0.65087,0.64657,0.6412,0.603],
[0.68954,0.68095,0.67772,0.67085,0.66805,0.65903,0.65087,0.6134],
[0.70243,0.69384,0.69061,0.68309,0.68095,0.66913,0.66161,0.621],
[0.71532,0.70673,0.70351,0.69599,0.69276,0.6788,0.67128,0.63066],
[0.72821,0.71962,0.7164,0.70888,0.70351,0.69061,0.68309,0.639],
[0.7411,0.73208,0.72714,0.72198,0.71747,0.70136,0.69276,0.649]
]

# --- Electrical Conductivity ---
ec_nonmag = [
[940,809,623,447,335,300,246,88.8],
[1056,892,706,523,384,350,303,103.4],
[1155,1031,809,599,447,408,348,116.7],
[1273,1158,918,693,517,462,393,128.9],
[1400,1294,1027,786,588,519,440,136.2],
[1537,1450,1130,862,654,567,486,146.5],
[1674,1519,1281,986,737,638,542,165.7],
[1810,1694,1445,1113,824,710,604,182.4],
[1972,1793,1564,1232,914,781,663,199.4]
]

ec_mag = [
[555,470,395,350,288,247,188,88.8],
[803,637,561,518,478,428,316,103.4],
[1000,736,660,617,577,558,415,116.7],
[1154,831,742,707,667,641,505,128.9],
[1244,922,851,816,776,750,614,136.2],
[1353,1042,971,926,896,870,734,146.5],
[1473,1125,1054,1009,979,950,817,165.7],
[1556,1285,1214,1169,1139,1101,977,182.4],
[1716,1445,1374,1329,1299,1261,1137,199.4]
]

# --- Viscosity ---
mu_nonmag = [
[1.501,1.461,1.391,1.341,1.291,1.261,1.231,1.134],
[1.33,1.29,1.22,1.17,1.12,1.09,1.06,1.01],
[1.21,1.11,1.032,0.9992,0.974,0.9512,0.895,0.88],
[1.08,1,0.982,0.961,0.942,0.917,0.8722,0.792],
[1.02,0.97,0.951,0.932,0.91,0.89,0.861,0.721],
[0.972,0.951,0.932,0.91,0.89,0.861,0.812,0.641],
[0.956,0.933,0.912,0.891,0.861,0.81,0.764,0.591],
[0.933,0.912,0.89,0.862,0.82,0.766,0.72,0.5421],
[0.933,0.912,0.89,0.862,0.82,0.766,0.72,0.5421]
]

mu_mag = [
[1.72,1.69,1.67,1.63,1.6,1.58,1.56,1.3059],
[1.39,1.36,1.34,1.3,1.27,1.25,1.23,1.134],
[1.22,1.19,1.17,1.13,1.1,1.09,1.06,1.01],
[1.09,1.06,1.04,1,0.97,0.95,0.92,0.88],
[1.01,0.98,0.96,0.92,0.89,0.87,0.84,0.792],
[0.93,0.9,0.88,0.84,0.81,0.79,0.76,0.721],
[0.88,0.85,0.83,0.79,0.76,0.74,0.71,0.641],
[0.83,0.8,0.78,0.74,0.71,0.69,0.66,0.591],
[0.78,0.75,0.73,0.69,0.66,0.64,0.61,0.5421]
]

# --- pH ---
ph_nonmag = [
[8.77,8.69,8.56,8.34,8.02,7.94,7.62,7.73],
[8.69,8.63,8.5,8.28,7.93,7.78,7.56,7.64],
[8.61,8.53,8.43,8.12,7.84,7.72,7.5,7.57],
[8.53,8.48,8.37,8.05,7.75,7.66,7.44,7.49],
[8.45,8.38,8.28,7.97,7.66,7.6,7.38,7.44],
[8.37,8.32,8.12,7.89,7.51,7.47,7.32,7.4],
[8.29,8.23,8.05,7.83,7.42,7.38,7.25,7.37],
[8.21,8.15,7.97,7.77,7.33,7.28,7.19,7.33],
[8.13,8.09,7.89,7.71,7.24,7.19,7.13,7.21]
]

ph_mag = [
[7.43,7.46,7.55,7.63,7.77,7.86,7.9,7.73],
[7.35,7.38,7.42,7.53,7.65,7.79,7.89,7.64],
[7.27,7.3,7.35,7.44,7.55,7.67,7.81,7.57],
[7.19,7.22,7.26,7.37,7.45,7.57,7.7,7.49],
[7.14,7.17,7.21,7.28,7.39,7.47,7.6,7.44],
[7.09,7.14,7.17,7.23,7.3,7.4,7.5,7.4],
[7.04,7.11,7.14,7.19,7.25,7.32,7.43,7.37],
[7.01,7.08,7.11,7.16,7.21,7.27,7.35,7.33],
[7.00,7.04,7.08,7.13,7.19,7.24,7.30,7.21]
]

# =========================================================
# 4. BUILD DATASET
# =========================================================

def build_dataset(k, ec, mu, ph, magnetic):
    X, Y = [], []
    for i, T in enumerate(temps):
        for j, phi in enumerate(concs):
            X.append([T, phi, magnetic, T*phi, phi**2])
            Y.append([k[i][j], ec[i][j], mu[i][j], ph[i][j]])
    return np.array(X), np.array(Y)

X1, Y1 = build_dataset(k_nonmag, ec_nonmag, mu_nonmag, ph_nonmag, 0)
X2, Y2 = build_dataset(k_mag, ec_mag, mu_mag, ph_mag, 1)

X = np.vstack((X1, X2))
Y = np.vstack((Y1, Y2))

# =========================================================
# 5. TRAIN MODEL
# =========================================================

scaler = StandardScaler()
X = scaler.fit_transform(X)

model = MultiOutputRegressor(
    RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42)
)

model.fit(X, Y)

# =========================================================
# 6. PREDICTION FUNCTION
# =========================================================

def predict_all(T, phi, magnetic):
    X_new = np.array([[T, phi, magnetic, T*phi, phi**2]])
    X_new = scaler.transform(X_new)
    pred = model.predict(X_new)[0]
    return pred

# =========================================================
# 7. GENERATE TABLE
# =========================================================

def generate_table(index, magnetic):
    table = []
    for T in temps:
        row = [T]
        for phi in concs:
            pred = predict_all(T, phi, magnetic)
            row.append(round(pred[index],4))
        table.append(row)
    return pd.DataFrame(table, columns=["Temp"]+[f"{c}" for c in concs])

# =========================================================
# 8. PLOT FUNCTION
# =========================================================

def plot_property(index, magnetic, title):
    plt.figure(figsize=(8,6))
    plt.gca().set_facecolor("#f0f0f0")

    markers = ['o','s','^','D','v','P','X','*']

    for i, phi in enumerate(concs):
        vals = []
        for T in temps:
            vals.append(predict_all(T, phi, magnetic)[index])

        plt.plot(temps, vals, marker=markers[i], linestyle='-', label=f"{phi}")

    plt.legend(loc="upper right")
    plt.xlabel("Temperature (°C)")
    plt.ylabel(title)
    plt.title(title)
    plt.grid(False)
    plt.show()

# =========================================================
# 9. OUTPUT
# =========================================================
# 9. EXTENDED OUTPUT
# =========================================================

# ---------- Fe3O4/Al2O3/TiO2 ----------
print("\n================ Fe3O4 / Al2O3 / TiO2 =================")

print("\nThermal Conductivity (k)")
print(generate_table(0,1))

print("\nElectrical Conductivity (EC)")
print(generate_table(1,1))

print("\nViscosity (μ)")
print(generate_table(2,1))

print("\npH Variation")
print(generate_table(3,1))


# ---------- Fe2O3/Al2O3/TiO2 ----------
print("\n================ Fe2O3 / Al2O3 / TiO2 =================")

print("\nThermal Conductivity (k)")
print(generate_table(0,0))

print("\nElectrical Conductivity (EC)")
print(generate_table(1,0))

print("\nViscosity (μ)")
print(generate_table(2,0))

print("\npH Variation")
print(generate_table(3,0))


# =========================================================
# 10. PLOTS
# =========================================================

# ---------- Fe3O4 Plots ----------
plot_property(0,1,"Thermal Conductivity (Fe3O4 / Al2O3 / TiO2)")
plot_property(1,1,"Electrical Conductivity (Fe3O4 / Al2O3 / TiO2)")
plot_property(2,1,"Viscosity (Fe3O4 / Al2O3 / TiO2)")
plot_property(3,1,"pH (Fe3O4 / Al2O3 / TiO2)")


# ---------- Fe2O3 Plots ----------
plot_property(0,0,"Thermal Conductivity (Fe2O3 / Al2O3 / TiO2)")
plot_property(1,0,"Electrical Conductivity (Fe2O3 / Al2O3 / TiO2)")
plot_property(2,0,"Viscosity (Fe2O3 / Al2O3 / TiO2)")
plot_property(3,0,"pH (Fe2O3 / Al2O3 / TiO2)")

# =========================================================
# 11. SAVE ALL OUTPUTS TO EXCEL
# =========================================================

with pd.ExcelWriter("Nanofluid_Predictions.xlsx", engine="openpyxl") as writer:

    # Fe3O4 / Al2O3 / TiO2
    generate_table(0,1).to_excel(writer, sheet_name="Fe3O4_k", index=False)
    generate_table(1,1).to_excel(writer, sheet_name="Fe3O4_EC", index=False)
    generate_table(2,1).to_excel(writer, sheet_name="Fe3O4_Viscosity", index=False)
    generate_table(3,1).to_excel(writer, sheet_name="Fe3O4_pH", index=False)

    # Fe2O3 / Al2O3 / TiO2
    generate_table(0,0).to_excel(writer, sheet_name="Fe2O3_k", index=False)
    generate_table(1,0).to_excel(writer, sheet_name="Fe2O3_EC", index=False)
    generate_table(2,0).to_excel(writer, sheet_name="Fe2O3_Viscosity", index=False)
    generate_table(3,0).to_excel(writer, sheet_name="Fe2O3_pH", index=False)

print("Excel file saved: Nanofluid_Predictions.xlsx")

# =========================================================
# 12. SAVE PLOTS
# =========================================================

def save_property_plot(index, magnetic, title, filename):

    plt.figure(figsize=(8,6))
    plt.gca().set_facecolor("#f0f0f0")

    markers = ['o','s','^','D','v','P','X','*']

    for i, phi in enumerate(concs):

        vals = []
        for T in temps:
            vals.append(predict_all(T, phi, magnetic)[index])

        plt.plot(
            temps,
            vals,
            marker=markers[i],
            linestyle='-',
            label=f"{phi}"
        )

    plt.xlabel("Temperature (°C)")
    plt.ylabel(title)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# Fe3O4
save_property_plot(0,1,"Thermal Conductivity","Fe3O4_k.png")
save_property_plot(1,1,"Electrical Conductivity","Fe3O4_EC.png")
save_property_plot(2,1,"Viscosity","Fe3O4_Viscosity.png")
save_property_plot(3,1,"pH","Fe3O4_pH.png")

# Fe2O3
save_property_plot(0,0,"Thermal Conductivity","Fe2O3_k.png")
save_property_plot(1,0,"Electrical Conductivity","Fe2O3_EC.png")
save_property_plot(2,0,"Viscosity","Fe2O3_Viscosity.png")
save_property_plot(3,0,"pH","Fe2O3_pH.png")

print("All figures saved as PNG files.")
