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

This project implements a linear regression model from scratch to predict car prices based on their mileage. The implementation uses gradient descent optimization with data normalization to ensure stable and accurate training. The trained model parameters are saved to a JSON file for future use.

## ✨ Features

- **🔧 From-scratch Implementation**: Complete linear regression implementation without external ML libraries
- **📊 Data Normalization**: Robust z-score normalization for stable training
- **🎯 Gradient Descent**: Efficient optimization with simultaneous parameter updates
- **🛡️ Data Validation**: Comprehensive CSV parsing with error handling
- **💾 Model Persistence**: Automatic saving of all model parameters and normalization data
- **📈 Training Visualization**: Real-time cost function monitoring during training
- **🔒 Robust Error Handling**: Graceful handling of malformed or invalid data

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

3. **No additional dependencies required** - uses only Python standard library!

## 💻 Usage

### Training the Model

1. **Prepare your data**: Ensure your CSV file is in the `data/` directory with the format `km,price`

2. **Run the training script**:
   ```bash
   python3 ft_linear_regresion.py
   ```

3. **View training progress**: The script will display:
   - Data loading and validation status
   - Training progress with cost function values
   - Final model parameters
   - Model save location

### Example Output
```
Loading and parsing data from data/data.csv...
✓ Data loaded successfully: 24 samples
✓ Data validation passed

Training linear regression model...
Iteration 100/10000 - Cost: 0.2847
Iteration 200/10000 - Cost: 0.1923
...
Iteration 10000/10000 - Cost: 0.0156

Training completed!
Final parameters:
- θ₀ (intercept): 8499.48
- θ₁ (slope): -0.0203
- Training cost: 0.0156

✓ Model saved to data/trained_model.json
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

## 📤 Output

### Trained Model File
The model is saved as `data/trained_model.json` containing:

```json
{
    "theta0": 8499.48,
    "theta1": -0.0203,
    "km_mean": 61637.24,
    "km_std": 61983.47,
    "price_mean": 8052.91,
    "price_std": 1597.03
}
```

### Parameters Explained
- **theta0/theta1**: Linear regression coefficients
- **km_mean/km_std**: Mileage normalization parameters
- **price_mean/price_std**: Price normalization parameters

## 🔧 Technical Implementation

### Key Functions

#### `open_and_parse(filename)`
- Validates CSV header format
- Parses numeric data with error handling
- Returns validated km and price lists

#### `normalize_data(data)`
- Implements z-score normalization
- Returns normalized data and statistics
- Handles edge cases (zero standard deviation)

#### `train_linear_regression(km_norm, price_norm, learning_rate, iterations)`
- Gradient descent optimization
- Simultaneous parameter updates
- Cost function monitoring
- Returns optimized parameters

#### `save_model(theta0, theta1, km_mean, km_std, price_mean, price_std)`
- Creates data directory if needed
- Saves all parameters to JSON
- Ensures data persistence

### Performance Characteristics
- **Time Complexity**: O(n × iterations) for training
- **Space Complexity**: O(n) for data storage
- **Default Settings**: 
  - Learning rate: 0.1
  - Iterations: 10,000
  - Convergence monitoring every 100 iterations

## 📁 Project Structure

```
ft_linear_regresion/
├── ft_linear_regresion.py    # Main training script
├── README.md                 # Project documentation
└── data/
    ├── data.csv             # Input training data
    └── trained_model.json   # Output model parameters
```

## 📋 Requirements

- **Python**: 3.6 or higher
- **Standard Library**: json, os (no external dependencies)
- **Input**: CSV file with km,price format
- **Output**: JSON file with model parameters

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is part of an educational exercise in machine learning fundamentals.

---

*Built with ❤️ using pure Python - no external ML libraries required!*