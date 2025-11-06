import pandas as pd
import random

# Step 1: Load your original CSV
df = pd.read_csv("/content/program_ratings.csv")

# Get all columns except the first ('Type of Program')
rating_columns = df.columns[1:]

# Step 2: Choose 5 random cells (row, column)
changes = []
for _ in range(5):
    row_idx = random.randint(0, len(df) - 1)
    col_name = random.choice(rating_columns)
    old_value = float(df.at[row_idx, col_name])
    variation = random.uniform(-0.3, 0.3)
    new_value = round(old_value + variation, 1)
    df.at[row_idx, col_name] = new_value
    changes.append((df.at[row_idx, 'Type of Program'], col_name, old_value, new_value))

# Step 3: Save the modified file
df.to_csv("program_ratings_modified.csv", index=False)

# Step 4: Show which cells were changed
print("Modified 5 random cells and saved as 'program_ratings_modified.csv'\n")
print("Changes made:")
for program, col, old, new in changes:
    print(f"Program: {program}, Column: {col}, Old: {old}, New: {new}")
