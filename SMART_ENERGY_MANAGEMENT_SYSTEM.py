import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, scrolledtext
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import random

# =================================================
# Helper utilities (Tkinter-friendly input)
# =================================================
def get_int_gui(prompt, min_val=None, max_val=None, valid=None):
    while True:
        try:
            val = simpledialog.askinteger("Input Required", prompt)
            if val is None:
                continue
            if valid is not None and val not in valid:
                messagebox.showerror("Error", f"Please enter one of {valid}")
                continue
            if min_val is not None and val < min_val:
                messagebox.showerror("Error", f"Enter value >= {min_val}")
                continue
            if max_val is not None and val > max_val:
                messagebox.showerror("Error", f"Enter value <= {max_val}")
                continue
            return val
        except:
            messagebox.showerror("Error", "Please enter a valid integer.")

def get_float_gui(prompt, min_val=None):
    while True:
        try:
            val = simpledialog.askfloat("Input Required", prompt)
            if val is None:
                continue
            if min_val is not None and val < min_val:
                messagebox.showerror("Error", f"Enter value >= {min_val}")
                continue
            return val
        except:
            messagebox.showerror("Error", "Please enter a valid number.")

def build_facility_names(n_res, n_ind, n_com):
    names = []
    for i in range(1, n_res+1):
        names.append(f"Residential{i}")
    for i in range(1, n_ind+1):
        names.append(f"Industrial{i}")
    for i in range(1, n_com+1):
        names.append(f"Commercial{i}")
    return names

# =================================================
# Energy cost model
# =================================================
def energy_cost(row, lighting_on=1, hvac_on=1):
    demand = lighting_on * row["Lighting"] + hvac_on * row["HVAC"]
    effective_consumption = max(row["TotalEnergy"] - row["LocalProd"], 0)
    return (effective_consumption + demand) * row["Price"]

# =================================================
# DRL Agent
# =================================================
class DRLAgent:
    def __init__(self, facility, price):
        self.facility = facility
        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.2
        self.price = price

    def get_state(self, row):
        return (round(row["LocalProd"]), round(row["Price"], 3))

    def choose_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros((2,2))
        if random.uniform(0,1) < self.epsilon:
            return (random.randint(0,1), random.randint(0,1))
        a_index = np.argmax(self.q_table[state])
        return (a_index // 2, a_index % 2)

    def update(self, state, action, reward, next_state):
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros((2,2))
        a_index = action[0]*2 + action[1]
        best_next = np.max(self.q_table[next_state])
        self.q_table[state].flat[a_index] += self.alpha * (
            reward + self.gamma*best_next - self.q_table[state].flat[a_index]
        )

# =================================================
# Main Execution wrapped in GUI
# =================================================
def run_program():
    data = {"Facility": [], "TotalEnergy": [], "Lighting": [],
            "HVAC": [], "Price": [], "LocalProd": []}

    # Facility counts
    n_res = get_int_gui("How many Residential facilities?", min_val=0)
    n_ind = get_int_gui("How many Industrial facilities?", min_val=0)
    n_com = get_int_gui("How many Commercial facilities?", min_val=0)

    facilities = build_facility_names(n_res, n_ind, n_com)
    data["Facility"] = facilities

    # Collect data
    for fac in facilities:
        te = get_float_gui(f"{fac} - TotalEnergy (kWh):", min_val=0.0)
        lt = get_float_gui(f"{fac} - Lighting (kWh):", min_val=0.0)
        hv = get_float_gui(f"{fac} - HVAC (kWh):", min_val=0.0)
        pr = get_float_gui(f"{fac} - Price ($/kWh):", min_val=0.0)
        lp = get_float_gui(f"{fac} - LocalProd (kWh):", min_val=0.0)
        data["TotalEnergy"].append(te)
        data["Lighting"].append(lt)
        data["HVAC"].append(hv)
        data["Price"].append(pr)
        data["LocalProd"].append(lp)

    df = pd.DataFrame(data)
    df["EnergyCost"] = df.apply(lambda r: energy_cost(r, 1, 1), axis=1)

    # Control mode
    mode = get_int_gui("Choose mode:\n1 = Full DRL\n2 = Human\n3 = Hybrid", valid={1,2,3})

    schedules = {}
    for idx, row in df.iterrows():
        agent = DRLAgent(row["Facility"], row["Price"])
        for _ in range(50):
            state = agent.get_state(row)
            act = agent.choose_action(state)
            cost = energy_cost(row, act[0], act[1])
            reward = -cost
            next_state = agent.get_state(row)
            agent.update(state, act, reward, next_state)

        if mode == 1:
            lighting_on, hvac_on = agent.choose_action(agent.get_state(row))
        elif mode == 2:
            lighting_on = get_int_gui(f"{row['Facility']} - Turn ON Lighting? (0/1):", valid={0,1})
            hvac_on = get_int_gui(f"{row['Facility']} - Turn ON HVAC? (0/1):", valid={0,1})
        else:
            hvac_on = get_int_gui(f"{row['Facility']} - Turn ON HVAC? (0/1):", valid={0,1})
            lighting_on, _ = agent.choose_action(agent.get_state(row))

        schedules[row["Facility"]] = {
            "Lighting_ON": lighting_on,
            "HVAC_ON": hvac_on,
            "OptimizedCost": energy_cost(row, lighting_on, hvac_on)
        }

    # Ensemble Learning
    X = df[["Lighting", "HVAC", "LocalProd"]].values
    y_true = df["TotalEnergy"].values

    models = {
        'lr': LinearRegression(),
        'dt': DecisionTreeRegressor(max_depth=3),
        'knn': KNeighborsRegressor(n_neighbors=2)
    }

    preds = {}
    for name, model in models.items():
        model.fit(X, y_true)
        preds[name] = model.predict(X)
    P = np.vstack([preds['lr'], preds['dt'], preds['knn']]).T

    def obj(w):
        return np.mean((y_true - P.dot(w))**2)

    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w)-1})
    bnds = [(0,1)]*3
    res = minimize(obj, x0=np.array([1/3,1/3,1/3]), bounds=bnds, constraints=cons)
    weights = res.x
    y_ens = P.dot(weights)

    # Results
    summary_rows = []
    for fac in df["Facility"]:
        sch = schedules[fac]
        baseline = df.loc[df["Facility"] == fac, "EnergyCost"].values[0]
        summary_rows.append({
            "Facility": fac,
            "Lighting_ON": sch["Lighting_ON"],
            "HVAC_ON": sch["HVAC_ON"],
            "OptimizedCost ($)": round(sch["OptimizedCost"], 4),
            "BaselineCost ($)": round(baseline, 4),
            "CostSavings ($)": round(baseline - sch["OptimizedCost"], 4)
        })
    summary_df = pd.DataFrame(summary_rows)

    ens_summary = pd.DataFrame({
        "Facility": df["Facility"],
        "TrueEnergy (kWh)": np.round(y_true, 4),
        "PredictedEnergy (kWh)": np.round(y_ens, 4)
    })

    unified_rows = []
    for i, fac in enumerate(df["Facility"]):
        sch = schedules[fac]
        baseline_cost = df.loc[df["Facility"] == fac, "EnergyCost"].values[0]
        optimized_cost = sch["OptimizedCost"]
        savings = baseline_cost - optimized_cost
        savings_pct = (savings / baseline_cost * 100) if baseline_cost > 0 else 0.0
        true_energy = df.loc[df["Facility"] == fac, "TotalEnergy"].values[0]
        pred_energy = y_ens[i]

        unified_rows.append({
            "Facility": fac,
            "Lighting_ON": sch["Lighting_ON"],
            "HVAC_ON": sch["HVAC_ON"],
            "BaselineCost ($)": round(baseline_cost, 4),
            "OptimizedCost ($)": round(optimized_cost, 4),
            "CostSavings ($)": round(savings, 4),
            "Savings % (Sim.)": round(savings_pct, 2),
            "TrueEnergy (kWh)": round(true_energy, 4),
            "PredEnergy (kWh)": round(pred_energy, 4)
        })
    unified_df = pd.DataFrame(unified_rows)

    # Show results in Tkinter window
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, "\n--- DRL Optimized Scheduling Table ---\n")
    result_text.insert(tk.END, summary_df.to_string(index=False))
    result_text.insert(tk.END, "\n\n--- Ensemble Learning Prediction Table ---\n")
    result_text.insert(tk.END, ens_summary.to_string(index=False))
    result_text.insert(tk.END, "\n\n--- Unified Optimization & Prediction Table ---\n")
    result_text.insert(tk.END, unified_df.to_string(index=False))

# =================================================
# Tkinter GUI Setup
# =================================================
root = tk.Tk()
root.title("Smart Energy Management System (Tkinter GUI)")
root.geometry("1000x700")

run_button = ttk.Button(root, text="Run Optimization", command=run_program)
run_button.pack(pady=10)

result_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=120, height=35)
result_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

root.mainloop()



