import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv("student_data.csv")

X = df[["participation", "assignments", "exams", "absences"]].values
y = df[["performance"]].values

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X = x_scaler.fit_transform(X)
y = y_scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


class RuleAwareANFIS(nn.Module):
    def __init__(self, n_inputs=4, n_mfs=3):
        super().__init__()

        self.n_inputs = n_inputs
        self.n_mfs = n_mfs

        self.input_names = [
            "participation",
            "assignments",
            "exams",
            "absences"
        ]

        
        self.rules = [
            {"participation": 2, "assignments": 2, "exams": 2, "absences": 0},
            {"participation": 2, "assignments": 2, "exams": 1, "absences": 0},
            {"participation": 1, "assignments": 2, "exams": 2, "absences": 0},
            {"participation": 2, "assignments": 1, "exams": 2, "absences": 1},
            {"participation": 1, "assignments": 1, "exams": 1, "absences": 1},
            {"participation": 0, "assignments": 1, "exams": 1, "absences": 2},
            {"participation": 0, "assignments": 0, "exams": 0, "absences": 2},
            {"participation": 2, "assignments": 0, "exams": 0, "absences": 0},
            {"participation": 1, "assignments": 2, "exams": 1, "absences": 1},
            {"participation": 0, "assignments": 2, "exams": 2, "absences": 0},
            {"exams": 2, "absences": 2},
            {"participation": 2, "absences": 2},
            {"assignments": 0, "exams": 2, "participation": 2},
            {"exams": 0, "absences": 2},
            {"participation": 2, "assignments": 2, "exams": 0, "absences": 0},
            {"participation": 1, "exams": 0, "assignments": 2, "absences": 0},
            {"participation": 0, "absences": 2},
            {"participation": 1, "assignments": 2, "exams": 2, "absences": 1},
            {"participation": 0, "assignments": 0, "exams": 0, "absences": 0},
            {"participation": 1, "assignments": 1, "exams": 1, "absences": 2},
            {"assignments": 0, "exams": 0},
        ]

        self.n_rules = len(self.rules)

        self.centers = nn.Parameter(torch.tensor([
            [0.0, 0.5, 1.0],  # συμμετοχη
            [0.0, 0.5, 1.0],  # εργασθες
            [0.0, 0.5, 1.0],  # εξετασεις
            [0.0, 0.5, 1.0],  # απουσιες
        ], dtype=torch.float32))

        self.sigmas = nn.Parameter(torch.ones(n_inputs, n_mfs) * 0.25)

        self.consequents = nn.Parameter(
            torch.randn(self.n_rules, n_inputs + 1) * 0.1
        )

    def gaussian_mf(self, x):
        x = x.unsqueeze(2)

        c = self.centers.unsqueeze(0)
        s = torch.clamp(self.sigmas.unsqueeze(0), min=1e-3)

        return torch.exp(-0.5 * ((x - c) / s) ** 2)

    def forward(self, x):
        batch_size = x.shape[0]

        mf_values = self.gaussian_mf(x)

        firing_strengths = []

        for rule in self.rules:
            strength = torch.ones(batch_size, device=x.device)

            for input_idx, input_name in enumerate(self.input_names):
                if input_name in rule:
                    mf_idx = rule[input_name]
                    strength *= mf_values[:, input_idx, mf_idx]

            firing_strengths.append(strength)

        firing_strengths = torch.stack(firing_strengths, dim=1)

        normalized_strengths = firing_strengths / (
            firing_strengths.sum(dim=1, keepdim=True) + 1e-8
        )

        x_aug = torch.cat(
            [x, torch.ones(batch_size, 1, device=x.device)],
            dim=1
        )

        rule_outputs = torch.matmul(x_aug, self.consequents.T)

        output = torch.sum(
            normalized_strengths * rule_outputs,
            dim=1,
            keepdim=True
        )

        return output


model = RuleAwareANFIS(n_inputs=4, n_mfs=3)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 1000

for epoch in range(epochs):
    model.train()

    optimizer.zero_grad()

    predictions = model(X_train)
    loss = criterion(predictions, y_train)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")


model.eval()

with torch.no_grad():
    y_pred = model(X_test).numpy()

y_test_original = y_scaler.inverse_transform(y_test.numpy())
y_pred_original = y_scaler.inverse_transform(y_pred)

mae = mean_absolute_error(y_test_original, y_pred_original)
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
r2 = r2_score(y_test_original, y_pred_original)

print("\nRULE-AWARE ANFIS Evaluation")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.3f}")


def predict_student(participation, assignments, exams, absences):
    sample = np.array([[participation, assignments, exams, absences]])

    sample_scaled = x_scaler.transform(sample)
    sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32)

    model.eval()

    with torch.no_grad():
        pred_scaled = model(sample_tensor).numpy()

    pred = y_scaler.inverse_transform(pred_scaled)[0][0]

    return float(np.clip(pred, 0, 100))


print("\nExamples")
print("Student A:", predict_student(75, 8, 9, 2))
print("Student B:", predict_student(30, 4, 3, 15))
print("Average student:", predict_student(50, 5, 5, 10))
print("Silent genius:", predict_student(10, 9, 10, 1))
print("Enthusiastic but struggling:", predict_student(90, 2, 2, 3))
print("Good grades but absent:", predict_student(20, 8, 8, 18))


print("\nNumber of fuzzy rules used:", model.n_rules)