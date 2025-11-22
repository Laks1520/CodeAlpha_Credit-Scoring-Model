import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

print("="*60)
print("CREDIT SCORING MODEL TRAINING")
print("="*60)

# STEP 1: Load the data
print("\n📂 Step 1: Loading data...")
data = pd.read_csv('data/credit_data.csv')
print(f"   Loaded {len(data)} records")

# STEP 2: Prepare the data
print("\n🔧 Step 2: Preparing data...")

# Separate features (X) and target (y)
X = data.drop('default', axis=1)  # Everything except 'default'
y = data['default']  # Just the 'default' column

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"   Training set: {len(X_train)} records")
print(f"   Testing set: {len(X_test)} records")

# STEP 3: Train different models
print("\n🤖 Step 3: Training models...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n   Training {name}...")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    print(f"   ✅ {name} Accuracy: {accuracy*100:.2f}%")
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    filename = f"models/{name.replace(' ', '_').lower()}.pkl"
    joblib.dump(model, filename)
    print(f"   💾 Saved to: {filename}")

# STEP 4: Compare models
print("\n📊 Step 4: Comparing models...")
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)
for name, accuracy in results.items():
    print(f"{name:25s}: {accuracy*100:.2f}%")

# Find best model
best_model_name = max(results, key=results.get)
best_accuracy = results[best_model_name]
print(f"\n🏆 Best Model: {best_model_name} ({best_accuracy*100:.2f}%)")

# STEP 5: Detailed evaluation of best model
print(f"\n📈 Step 5: Detailed evaluation of {best_model_name}...")

best_model = models[best_model_name]
y_pred = best_model.predict(X_test)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=['No Default', 'Default']))

# STEP 6: Create visualizations
print("\n📊 Step 6: Creating visualizations...")
os.makedirs('results', exist_ok=True)

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Default', 'Default'],
            yticklabels=['No Default', 'Default'])
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('results/confusion_matrix.png')
print("   ✅ Saved: results/confusion_matrix.png")

# Model Comparison
plt.figure(figsize=(10, 6))
model_names = list(results.keys())
accuracies = list(results.values())
colors = ['#FF6B6B' if acc < 0.7 else '#4ECDC4' if acc < 0.8 else '#95E1D3' 
          for acc in accuracies]
plt.bar(model_names, accuracies, color=colors, edgecolor='black', linewidth=1.5)
plt.ylabel('Accuracy')
plt.title('Model Performance Comparison')
plt.ylim([0, 1])
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.02, f'{v*100:.1f}%', ha='center', fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('results/model_comparison.png')
print("   ✅ Saved: results/model_comparison.png")

# Feature Importance (for Random Forest)
if best_model_name == 'Random Forest':
    plt.figure(figsize=(10, 6))
    importances = best_model.feature_importances_
    feature_names = X.columns
    indices = np.argsort(importances)[::-1]
    
    plt.bar(range(len(importances)), importances[indices], color='teal')
    plt.xticks(range(len(importances)), 
               [feature_names[i] for i in indices], 
               rotation=45, ha='right')
    plt.ylabel('Importance')
    plt.title('Feature Importance - What Matters Most?')
    plt.tight_layout()
    plt.savefig('results/feature_importance.png')
    print("   ✅ Saved: results/feature_importance.png")

print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)
print("\nNext step: Run 'python src/predict_simple.py' to make predictions")
