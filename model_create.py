import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Step 1: Load JSON data with explicit 'utf-8' encoding
with open('ConsumeWise.final_merged_data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Step 2: Convert JSON to DataFrame
df = pd.json_normalize(data, max_level=1)

# Step 3: Remove unnecessary columns
df_cleaned = df.drop(columns=['PMID Count', '_id.$oid'], errors='ignore')

# Step 4: Check available columns
print("Available columns:", df_cleaned.columns.tolist())

# Step 5: Check the contents of 'evidence'
print("\nFirst few entries in 'evidence':")
print(df_cleaned['evidence'].head())

# Step 6: Convert annotations lists to strings for unique checking
df_cleaned['annotations_as_str'] = df_cleaned['annotations'].apply(lambda x: str(x) if isinstance(x, list) else x)

# Step 7: Check unique values in 'annotations'
print("\nUnique values in 'annotations':")
print(df_cleaned['annotations_as_str'].unique())

# Step 8: Define a function to extract average evidence score
def extract_scores(evidence):
    if isinstance(evidence, list):
        scores = [item.get('score', 0) for item in evidence if isinstance(item, dict)]
        return sum(scores) / len(scores) if scores else 0
    return 0

# Step 9: Calculate average evidence score and add it to the DataFrame
df_cleaned['average_evidence_score'] = df_cleaned['evidence'].apply(extract_scores)

# Step 10: Display the updated DataFrame with the new column
print("\nDataFrame with average evidence score:")
print(df_cleaned[['evidence', 'average_evidence_score']].head())

# Step 11: Save the cleaned DataFrame to a new CSV file
csv_file_path = 'cleaned_consume_wise_data.csv'
df_cleaned.to_csv(csv_file_path, index=False)
print(f"\nCleaned data has been saved to '{csv_file_path}'.")

# Step 12: Prepare data for model training
# For demonstration, let's assume 'Type' is the target variable and we want to predict it
X = df_cleaned[['average_evidence_score']]  # Features
y = df_cleaned['Type']  # Target variable

# Step 13: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 14: Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 15: Make predictions
y_pred = model.predict(X_test)

# Step 16: Evaluate the model
print("\nModel Evaluation:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 17: Save the trained model
model_file_path = 'trained_model.pkl'
joblib.dump(model, model_file_path)
print(f"\nTrained model has been saved to '{model_file_path}'.")
