from ortools.sat.python import cp_model
import pandas as pd
import joblib

# ===========================
# Load trained ML model
# ===========================
model = joblib.load("delay_predictor.pkl")

# ===========================
# Load dataset
# ===========================
df = pd.read_csv("synthetic_trains.csv").head(10)  # demo with 10 trains

print("CSV Columns:", df.columns.tolist())
print("\nSample Data:\n", df.head())

# ===========================
# Encode categorical columns (if not already encoded)
# ===========================
if "train_type_enc" not in df.columns:
    df["train_type_enc"] = df["train_type"].map({"Express": 0, "Local": 1, "Freight": 2})

if "weather_enc" not in df.columns and "weather" in df.columns:
    df["weather_enc"] = df["weather"].map({"Sunny": 0, "Rainy": 1, "Storm": 2})

# ===========================
# Predict delays
# ===========================
features = ["train_type_enc", "time_of_day", "day_of_week",
            "weather_enc", "congestion", "historical_delay"]

df["predicted_delay"] = model.predict(df[features])

print("\nAvailable columns after encoding:", df.columns.tolist())
print("\nSample with predicted delays:\n", df.head())

# ===========================
# OR-Tools Optimization
# ===========================
cp = cp_model.CpModel()

horizon = 500  # planning horizon (minutes)
start_vars = {}

# Decision variables: start time for each train
for idx, row in df.iterrows():
    start_vars[row["train_id"]] = cp.NewIntVar(0, horizon, f"start_{row['train_id']}")

# Constraint: at least 5 minutes headway between trains
headway = 5
for i in range(len(df)):
    for j in range(i + 1, len(df)):
        t1, t2 = df.iloc[i]["train_id"], df.iloc[j]["train_id"]
        cp.Add(start_vars[t1] + headway <= start_vars[t2]).OnlyEnforceIf(
            start_vars[t1] <= start_vars[t2]
        )
        cp.Add(start_vars[t2] + headway <= start_vars[t1]).OnlyEnforceIf(
            start_vars[t2] <= start_vars[t1]
        )

# Objective: minimize weighted delay
weights = {"Express": 5, "Local": 3, "Freight": 1}
objective_terms = []

for idx, row in df.iterrows():
    delay = int(row["predicted_delay"])
    w = weights.get(row["train_type"], 2)  # default weight=2
    objective_terms.append(start_vars[row["train_id"]] + delay * w)

cp.Minimize(sum(objective_terms))

# ===========================
# Solve
# ===========================
solver = cp_model.CpSolver()
status = solver.Solve(cp)

if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    print("\n✅ Optimized Train Schedule:")
    for idx, row in df.iterrows():
        print(f"Train {row['train_id']} ({row['train_type']}) -> "
              f"Start: {solver.Value(start_vars[row['train_id']])} min, "
              f"Predicted delay: {row['predicted_delay']} min")
else:
    print("❌ No feasible solution found.")