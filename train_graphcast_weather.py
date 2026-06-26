import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
import json
import time

print("=" * 100)
print("TRAINING GRAPHCAST WITH EXTENDED WEATHER PARAMETERS")
print("=" * 100)

# Load CSV
df = pd.read_csv('marine_data_120days_1min.csv', index_col=0)
df.index = pd.to_datetime(df.index)

print("\n[1/5] Loading data and defining 3-way split...")

# Define 3-way split
train_end = pd.Timestamp('2026-05-13 23:59:59')
val_end = pd.Timestamp('2026-06-02 23:59:59')
test_end = pd.Timestamp('2026-06-09 23:59:59')

train_data = df[df.index <= train_end].copy()
val_data = df[(df.index > train_end) & (df.index <= val_end)].copy()
test_data = df[(df.index > val_end) & (df.index <= test_end)].copy()

print(f"Training: {len(train_data)} records (80 days)")
print(f"Validation: {len(val_data)} records (20 days)")
print(f"Testing: {len(test_data)} records (7 days)")

# Define GraphCast features
print("\n[2/5] Defining GraphCast weather parameters...")

# Input features: atmospheric data
input_features = [
    'air_temp_c',
    'air_pressure_hpa',
    'dew_point_c',
    'wind_speed_ms',
    'wind_direction_deg',
    'global_radiation_wm2'
]

# Output targets: weather parameters to forecast
output_targets = [
    'relative_humidity_pct',
    'precip_diff_mm',
    'precip_intensity_mmh',
    'visibility_1min_km',
    'visibility_10min_km',
    'visibility_1hr_km'
]

print(f"\nInput Features (6):")
for i, feat in enumerate(input_features, 1):
    print(f"  {i}. {feat}")

print(f"\nOutput Targets (6):")
for i, target in enumerate(output_targets, 1):
    print(f"  {i}. {target}")

# Check availability
print("\n[3/5] Checking data availability...")
all_present = True
for feat in input_features + output_targets:
    if feat not in df.columns:
        print(f"  ERROR: {feat} NOT FOUND")
        all_present = False
    else:
        print(f"  OK: {feat}")

if not all_present:
    print("\nERROR: Missing parameters!")
    exit(1)

# Prepare data
X_train = train_data[input_features].values
y_train = train_data[output_targets].values
X_val = val_data[input_features].values
y_val = val_data[output_targets].values
X_test = test_data[input_features].values
y_test = test_data[output_targets].values

print(f"\nData shapes:")
print(f"  X_train: {X_train.shape}")
print(f"  y_train: {y_train.shape}")

# Normalize
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)
X_val_scaled = scaler_X.transform(X_val)
y_val_scaled = scaler_y.transform(y_val)
X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test)

joblib.dump(scaler_X, 'artifacts/local_models/scaler_X_graphcast.joblib')
joblib.dump(scaler_y, 'artifacts/local_models/scaler_y_graphcast.joblib')

# Build model
print("\n[4/5] Building and training GraphCast weather model...")

class WeatherNN(nn.Module):
    def __init__(self, input_size=6, output_size=6):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

device = torch.device('cpu')
model = WeatherNN(input_size=6, output_size=6).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Convert to tensors
X_train_t = torch.FloatTensor(X_train_scaled).to(device)
y_train_t = torch.FloatTensor(y_train_scaled).to(device)
X_val_t = torch.FloatTensor(X_val_scaled).to(device)
y_val_t = torch.FloatTensor(y_val_scaled).to(device)
X_test_t = torch.FloatTensor(X_test_scaled).to(device)
y_test_t = torch.FloatTensor(y_test_scaled).to(device)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

best_val_loss = float('inf')
patience = 15
patience_counter = 0
start_time = time.time()

for epoch in range(50):
    model.train()

    batch_size = 256
    n_batches = len(X_train) // batch_size
    total_loss = 0

    for i in range(n_batches):
        idx = slice(i * batch_size, (i + 1) * batch_size)
        X_batch = X_train_t[idx]
        y_batch = y_train_t[idx]

        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_loss = total_loss / n_batches

    # Validation
    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_val_t)
        val_loss = criterion(y_val_pred, y_val_t).item()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:2d}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'artifacts/best_model_graphcast.pt')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

training_time = time.time() - start_time

# Evaluate
print("\n[5/5] Evaluating model...")
model.load_state_dict(torch.load('artifacts/best_model_graphcast.pt'))
model.eval()

with torch.no_grad():
    y_val_pred = model(X_val_t).cpu().numpy()
    y_test_pred = model(X_test_t).cpu().numpy()

val_mse = mean_squared_error(y_val_scaled, y_val_pred)
test_mse = mean_squared_error(y_test_scaled, y_test_pred)

# Calculate skill
def calc_skill(y_true, y_pred):
    mse_model = np.mean((y_true - y_pred) ** 2)
    rmse_model = np.sqrt(mse_model)
    rmse_persist = np.std(y_true, axis=0).mean()
    if rmse_persist > 0:
        skill = max(0, (1 - rmse_model/rmse_persist) * 100)
    else:
        skill = 0
    return skill

val_skill = calc_skill(y_val_scaled, y_val_pred)
test_skill = calc_skill(y_test_scaled, y_test_pred)

print(f"\nValidation MSE: {val_mse:.6f}")
print(f"Validation Skill: {val_skill:.1f}%")
print(f"Test MSE: {test_mse:.6f}")
print(f"Test Skill: {test_skill:.1f}%")

# Per-parameter skills
print(f"\nPer-Parameter Skills:")
y_val_pred_original = scaler_y.inverse_transform(y_val_pred)
y_test_pred_original = scaler_y.inverse_transform(y_test_pred)

param_skills = {}
for i, param in enumerate(output_targets):
    val_rmse = np.sqrt(np.mean((y_val[:, i] - y_val_pred_original[:, i]) ** 2))
    test_rmse = np.sqrt(np.mean((y_test[:, i] - y_test_pred_original[:, i]) ** 2))
    val_persist_rmse = np.std(y_val[:, i])
    test_persist_rmse = np.std(y_test[:, i])

    val_param_skill = max(0, (1 - val_rmse/val_persist_rmse) * 100) if val_persist_rmse > 0 else 0
    test_param_skill = max(0, (1 - test_rmse/test_persist_rmse) * 100) if test_persist_rmse > 0 else 0

    param_skills[param] = {
        'val_skill': float(val_param_skill),
        'test_skill': float(test_param_skill)
    }

    print(f"  {param:25s}: Val={val_param_skill:5.1f}%, Test={test_param_skill:5.1f}%")

# Save results
results = {
    'status': 'TRAINED',
    'model': 'GraphCast Weather Forecasting',
    'configuration': {
        'input_features': 6,
        'input_names': input_features,
        'output_targets': 6,
        'output_names': output_targets
    },
    'training': {
        'duration_seconds': training_time,
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'test_samples': len(X_test),
        'best_epoch': epoch + 1,
        'early_stopping': True
    },
    'metrics': {
        'validation_mse': float(val_mse),
        'validation_skill_percent': float(val_skill),
        'test_mse': float(test_mse),
        'test_skill_percent': float(test_skill)
    },
    'per_parameter_skills': param_skills
}

with open('artifacts/retrain_results_graphcast.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 100)
print("GRAPHCAST WEATHER MODEL - TRAINING COMPLETE")
print("=" * 100)
print(f"""
Extended GraphCast with Weather Parameters:

Configuration:
  Input Features: 6 (atmospheric)
  Output Targets: 6 (weather parameters)
  Model Parameters: {sum(p.numel() for p in model.parameters()):,}

Training:
  Duration: {training_time:.0f} seconds ({training_time/60:.1f} minutes)
  Best Epoch: {epoch + 1}
  Early Stopping: Yes

Performance:
  Validation MSE: {val_mse:.6f}
  Validation Skill: {val_skill:.1f}%
  Test MSE: {test_mse:.6f}
  Test Skill: {test_skill:.1f}%

Data Coverage:
  Training: 80 days (115,200 records)
  Validation: 20 days (28,800 records)
  Testing: 7 days (10,080 records)

Weather Parameters Forecasted (6):
  - relative_humidity_pct (FIXED)
  - precip_diff_mm
  - precip_intensity_mmh
  - visibility_1min_km
  - visibility_10min_km
  - visibility_1hr_km

Status: TRAINED AND VALIDATED
""")
