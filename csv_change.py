import pandas as pd

# Read the CSV file
df = pd.read_csv("/media/anjali/Data/home/guest/Documents/Zuruki/test_predictions.csv")

# Round each value in the label column to the nearest integer
df["label"] = df["label"].round().astype("Int64")

# Save the updated CSV file
df.to_csv("output.csv", index=False)

print("Label column rounded and saved to output.csv")