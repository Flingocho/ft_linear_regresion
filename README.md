# 🚗 Linear Regression Car Price Predictor

A robust and efficient linear regression training engine that predicts car prices based on mileage using gradient descent optimization with proper data normalization.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Algorithm Details](#algorithm-details)
- [Data Format](#data-format)
- [Output](#output)
- [Technical Implementation](#technical-implementation)
- [Project Structure](#project-structure)
- [Requirements](#requirements)

## 🎯 Overview

This project implements a complete machine learning pipeline from scratch to predict car prices based on their mileage. The implementation uses gradient descent optimization with data normalization for stable training, and includes both training and prediction capabilities with beautiful interactive visualizations.

![Sample Visualization](images/sample.png)

*Beautiful interactive visualization with dark theme, hover tooltips, and highlighted predictions*

## ✨ Features

### 🔧 Core ML Implementation
- **From-scratch Implementation**: Complete linear regression without external ML libraries
- **📊 Data Normalization**: Robust z-score normalization for stable training
- **🎯 Gradient Descent**: Efficient optimization with simultaneous parameter updates
- **🛡️ Data Validation**: Comprehensive CSV parsing with error handling
- **💾 Model Persistence**: Automatic saving of all model parameters and normalization data

### 🖥️ Training Features
- **📈 Training Visualization**: Real-time cost function monitoring during training
- **🎨 Interactive Training Plots**: Beautiful dark-themed matplotlib visualization
- **🖱️ Hover Functionality**: Interactive tooltips for data points and regression line
- **📊 Extended Regression Line**: Line extends beyond data points for clear trend visualization

### 💰 Prediction Features  
- **🧮 Interactive Calculator**: User-friendly price prediction interface
- **🔄 Robust Model Loading**: Graceful fallback to default values on errors
- **📊 Prediction Visualization**: Highlight predictions on training data graph
- **⭐ Enhanced Display**: Prediction points shown as large stars, training data dimmed
- **🖱️ Full Interactivity**: Hover tooltips for training data, predictions, and regression line
- **🔒 Error Handling**: Comprehensive validation and user input handling
- **📈 Training Visualization**: Real-time cost function monitoring during training
- **� Data Visualization**: Optional matplotlib-based plotting of data and regression line
- **�🔒 Robust Error Handling**: Graceful handling of malformed or invalid data

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ft_linear_regresion
   ```

2. **Ensure Python 3.6+ is installed**:
   ```bash
   python3 --version
   ```

3. **Install optional dependencies for visualization**:
   ```bash
   pip install matplotlib  # For data visualization (optional)
   ```

4. **No additional dependencies required for core functionality** - uses only Python standard library!

## 💻 Usage

### Training the Model

1. **Prepare your data**: Ensure your CSV file is in the `data/` directory with the format `km,price`

2. **Run the training script**:
   ```bash
   python3 train_model.py
   ```

3. **View training progress**: The script will display:
   - Data loading and validation status
   - Training progress with cost function values
   - Final model parameters
   - Model save location

4. **Interactive visualization**: After training, you can choose to visualize:
   - Original data points as gradient scatter plot
   - Interactive regression line with mouse hover
   - Real-time price predictions as you move your mouse
   - Modern dark theme with professional styling

### Making Predictions

1. **Run the prediction script**:
   ```bash
   python3 predict_price.py
   ```

2. **Enter car mileage**: Input the kilometers to get price predictions

3. **Interactive prediction visualization**: After each prediction, you can choose to:
   - See the prediction highlighted on the graph as a large star
   - View training data as smaller, semi-transparent points
   - Interactive hover tooltips for all elements
   - Same beautiful dark theme as training visualization

### Training Example Output
```
🎯 Training linear regression model...
   Learning rate: 0.1
   Iterations: 1000
   📊 Normalizing data...
   Mileage: mean=101066.25, std=51565.19
   Prices:  mean=6331.83, std=1291.87

   Iteration  200: Cost =   0.500000, θ0 = -0.050000, θ1 = -0.800000
   Iteration  400: Cost =   0.450000, θ0 = -0.020000, θ1 = -0.820000
   ...

✅ Training completed!
   Final θ0 (normalized): 0.000000
   Final θ1 (normalized): -0.856139
   Final cost: 0.123456

💾 Model parameters saved to 'data/trained_model.json'

🎨 Do you want to visualize the data and regression line? (y/n)
```

### Prediction Example Output
```
🚗 CAR PRICE PREDICTOR 💰
Predict prices using trained model

📁 Loading trained model...
✅ Model loaded successfully from 'data/trained_model.json'
   θ0: 0.000000
   θ1: -0.856139
   Normalization parameters loaded

🔢 Car Price Calculator
   Enter mileage to get price prediction
   Type 'quit' or 'exit' to stop

🚗 Enter car mileage (km): 150000
💰 Predicted price: $5,247.33
   Model: price = 0.000 + -0.856 × normalized_mileage

📊 Do you want to see this prediction on the graph? (y/n): y
📈 Loading training data for visualization...
🎨 Creating visualization...
[Interactive graph opens with prediction highlighted]
```
📊 Creating beautiful dark theme visualization...
✅ Beautiful dark theme visualization completed!
```

## 🧮 Algorithm Details

### Linear Regression Model
The model implements the hypothesis:
```
h(x) = θ₀ + θ₁ × x
```

Where:
- `θ₀` (theta0): y-intercept
- `θ₁` (theta1): slope coefficient
- `x`: normalized mileage
- `h(x)`: predicted normalized price

### Cost Function
Uses Mean Squared Error (MSE):
```
J(θ₀,θ₁) = 1/(2m) × Σ(h(xᵢ) - yᵢ)²
```

### Gradient Descent
Simultaneous parameter updates:
```
θ₀ := θ₀ - α × (1/m) × Σ(h(xᵢ) - yᵢ)
θ₁ := θ₁ - α × (1/m) × Σ((h(xᵢ) - yᵢ) × xᵢ)
```

### Data Normalization
Z-score normalization for both features and targets:
```
x_normalized = (x - μ) / σ
```

## 📊 Data Format

### Input CSV Requirements
- **Header**: Must be exactly `km,price`
- **Format**: Comma-separated values
- **Data Types**: Numeric values only
- **Example**:
  ```csv
  km,price
  240000,3650
  139800,3800
  150500,4400
  185530,4450
  ```

### Data Validation
- Validates header format
- Checks for numeric data types
- Handles missing or malformed entries
- Provides detailed error messages

## �️ Sample Visualization

Here's what the interactive visualization looks like with the training data and a prediction:

![Sample Visualization](images/sample.png)

*Interactive dark-themed visualization showing:*
- **Training data points** (gradient-colored by price)
- **Regression line** (blue, extends beyond data)
- **Prediction point** (large pink star)
- **Hover tooltips** for all elements
- **Beautiful dark theme** with professional styling

## �📤 Output

### Trained Model File
The model is saved as `data/trained_model.json` containing:

```json
{
    "theta0": 0.000000,
    "theta1": -0.856139,
    "mileage_mean": 101066.25,
    "mileage_std": 51565.19,
    "price_mean": 6331.83,
    "price_std": 1291.87
}
```

### Parameters Explained
- **theta0/theta1**: Linear regression coefficients
- **mileage_mean/mileage_std**: Mileage normalization parameters
- **price_mean/price_std**: Price normalization parameters

## 🔧 Technical Implementation

### Training Functions (train_model.py)

#### `open_and_parse(filename)`
- Validates CSV header format
- Parses numeric data with error handling
- Returns validated mileage and price lists

#### `normalize_data(data)`
- Implements z-score normalization
- Returns normalized data and statistics
- Handles edge cases (zero standard deviation)

#### `train_linear_regression(mileage, prices, learning_rate, iterations)`
- Gradient descent optimization
- Simultaneous parameter updates
- Cost function monitoring
- Returns optimized parameters

#### `save_model(theta0, theta1, mileage_mean, mileage_std, price_mean, price_std)`
- Creates data directory if needed
- Saves all parameters to JSON
- Ensures data persistence

#### `visualize_regression(mileage, prices, theta0, theta1, mileage_mean, mileage_std, price_mean, price_std)`
- Creates beautiful interactive dark theme visualization
- Gradient scatter plot with color-coded prices
- Interactive regression line with mouse hover functionality
- Real-time price predictions displayed as tooltips
- Modern dark theme with professional styling
- Handles matplotlib import gracefully

### Prediction Functions (predict_price.py)

#### `load_model(filename)`
- Loads trained model parameters from JSON
- Validates all required keys exist
- Returns default values (0, 0) on any error
- Comprehensive error handling for file/format issues

#### `predict_price(mileage, theta0, theta1, mileage_mean, mileage_std, price_mean, price_std)`
- Normalizes input mileage using training statistics
- Applies linear regression model
- Denormalizes result to original price scale
- Returns predicted price

#### `load_training_data(filename)`
- Loads original training data for visualization
- Validates CSV format and data integrity
- Returns mileage and price lists for plotting

#### `visualize_prediction(mileage, prices, theta0, theta1, mileage_mean, mileage_std, price_mean, price_std, pred_mileage, pred_price)`
- Shows training data as smaller, semi-transparent points
- Highlights prediction as large star marker
- Interactive hover tooltips for all elements
- Adapts axis limits to include prediction point
- Same beautiful dark theme as training visualization

### Performance Characteristics
- **Time Complexity**: O(n × iterations) for training, O(1) for prediction
- **Space Complexity**: O(n) for data storage
- **Default Settings**: 
  - Learning rate: 0.1
  - Iterations: 1,000
  - Convergence monitoring every 200 iterations

## 📁 Project Structure

```
ft_linear_regresion/
├── train_model.py           # Main training script
├── predict_price.py         # Interactive price prediction calculator
├── README.md                # Complete project documentation
└── data/
    ├── data.csv             # Input training data
    └── trained_model.json   # Output model parameters
```

## 📋 Requirements

### System Requirements
- **Python**: 3.6 or higher
- **Operating System**: Linux, macOS, Windows

### Dependencies
- **Core Dependencies**: numpy, json, os (included in Python standard library)
- **Optional Dependencies**: matplotlib (for beautiful interactive visualizations)

### Input/Output
- **Input**: CSV file with km,price format
- **Output**: JSON file with model parameters
- **Visualization**: Interactive matplotlib plots (optional)

### Installation Commands
```bash
# Core functionality (no extra dependencies needed)
python3 train_model.py

# For visualization features
pip install matplotlib numpy

# Run prediction calculator
python3 predict_price.py
```

## 🎯 Quick Start

1. **Train the model**:
   ```bash
   python3 train_model.py
   ```

2. **Make predictions**:
   ```bash
   python3 predict_price.py
   ```

3. **Enjoy beautiful visualizations** with interactive hover tooltips!

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is part of an educational exercise in machine learning fundamentals.

---

<div align="center">
  Created with ❤️ by <a href="https://github.com/Flingocho">Flingocho</a>
</div>