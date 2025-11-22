# Credit Scoring Model

Predict if someone will repay a loan or not using their financial history.

## What This Does
- Predicts if a person will default on a loan (YES/NO)
- Uses past financial data like income, debts, payment history
- Helps banks make better lending decisions

## How to Use

### Step 1: Install Python Libraries
```bash
pip install -r requirements.txt
```

### Step 2: Generate Sample Data
```bash
python src/generate_data.py
```

### Step 3: Train the Model
```bash
python src/train_simple.py
```

### Step 4: Test Predictions
```bash
python src/predict_simple.py
```

## What You'll Learn
- How to predict credit risk
- Feature engineering (creating useful data from raw data)
- Comparing different ML models
- Evaluating model performance

## Files Structure
```
credit-scoring-model/
├── data/              # Where data is stored
├── models/            # Where trained models are saved
├── results/           # Graphs and results
├── src/               # All Python code
│   ├── generate_data.py       # Creates fake credit data
│   ├── train_simple.py        # Trains the model
│   └── predict_simple.py      # Makes predictions
└── requirements.txt   # Libraries needed
```

## Results You'll Get
- Model accuracy (how often it's correct)
- Confusion matrix (what it got right/wrong)
- Feature importance (what matters most)
- Comparison of different models

## Common Terms
- **Default**: When someone doesn't repay their loan
- **Features**: Information about a person (income, age, etc.)
- **Training**: Teaching the computer to recognize patterns
- **Prediction**: Guessing if someone will default or not
