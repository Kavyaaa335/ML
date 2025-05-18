import pandas as pd
data = {
    "Weather": ["Sunny", "Sunny", "Rainy", "Sunny"],
    "Temperature": ["Warm", "Warm", "Cold", "Warm"],
    "Humidity": ["Normal", "High", "High", "High"],
    "Wind": ["Strong", "Strong", "Strong", "Weak"],
    "PlayTennis": ["Yes", "Yes", "No", "Yes"]
}

df = pd.DataFrame(data)
file_path = "training_data.csv"
df.to_csv(file_path, index=False)
print(f"CSV file 'training_data.csv' has been created successfully!")
def find_s_algorithm(data):
      features = data.iloc[:, :-1].values
      labels = data.iloc[:, -1].values
      hypothesis = None
      for i, label in enumerate(labels):
          if label == "Yes":
              hypothesis = features[i].copy()
              break
      if hypothesis is None:
          return "No positive examples found."

      for i, label in enumerate(labels):
          if label == "Yes":
              for j in range(len(hypothesis)):
                  if hypothesis[j] != features[i][j]:
                      hypothesis[j] = '?'
      return hypothesis


file_path = "training_data.csv"
data = pd.read_csv(file_path)
print("Training Data:")
print(data)
final_hypothesis = find_s_algorithm(data)
print("\nFinal Hypothesis:")
print(final_hypothesis)
