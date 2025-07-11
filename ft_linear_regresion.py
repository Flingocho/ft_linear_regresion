import numpy as np
import json

def open_and_parse():
    """Open the file in data/data.csv and parse the content"""
    # Open the file with read permission and save it in file
    with open("data/data.csv", "r") as file:
        content = file.read()
    
    # Split by lines first, then by commas
    lines = content.strip().split('\n')  # Split into lines
    
    # Separate lists for kilometers and prices
    kilometros = []
    precios = []
    
    # Validate that the first line is exactly "km,price"
    if not lines:
        print("❌ Error: Empty file")
        return None, None
    
    expected_header = "km,price"
    actual_header = lines[0].strip()
    
    if actual_header != expected_header:
        print(f"❌ Error: Invalid header")
        print(f"Expected: '{expected_header}'")
        print(f"Found: '{actual_header}'")
        return None, None
    
    print("✅ Valid header detected:", actual_header)
    
    # Process the lines starting from line 1 (skip header)
    start_index = 1
    
    for i in range(start_index, len(lines)):
        line = lines[i]
        row = line.split(',')  # Split each line by commas
        
        # Save kilometers and prices in separate lists
        if len(row) >= 2:
            try:
                kilometros.append(float(row[0]))  # Convert to number
                precios.append(float(row[1]))     # Convert to number
            except ValueError:
                print(f"❌ Error converting line {i+1}: {row}")
                return None, None
            
            if kilometros and precios and (kilometros[-1] < 0 or precios[-1] < 0):
                # Line number to show to user (i+1 because we always have header)
                print(f"❌ Error: negative value in row: {i+1}")
                print(f"      -> {lines[i]}")
                return None, None

    print(f"🧮 Total training points: {len(kilometros)} points")
    
    return kilometros, precios  # Return both lists separately


def normalize_data(data):
    """Normalize data to prevent value explosion during training"""
    data_array = np.array(data)
    mean = np.mean(data_array)
    std = np.std(data_array)
    
    # Prevent division by zero
    if std == 0:
        std = 1
    
    normalized = (data_array - mean) / std
    return normalized, mean, std

def denormalize_price(normalized_price, price_mean, price_std):
    """Convert normalized price back to original scale"""
    return normalized_price * price_std + price_mean

def estimate_price_normalized(normalized_mileage, theta0, theta1):
    """Calculate estimated normalized price using linear regression formula"""
    return theta0 + theta1 * normalized_mileage

def train_linear_regression(kilometros, precios, learning_rate=0.01, iterations=1000):
    """
    Train linear regression model using gradient descent with normalization
    Formula: tmpθ0 = learningRate * (1/m) * Σ(estimatePrice(mileage[i]) - price[i])
             tmpθ1 = learningRate * (1/m) * Σ((estimatePrice(mileage[i]) - price[i]) * mileage[i])
    """
    print(f"\n🎯 Training linear regression model...")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Iterations: {iterations}")
    
    # Normalize the data to prevent explosion
    print(f"   📊 Normalizing data...")
    normalized_km, km_mean, km_std = normalize_data(kilometros)
    normalized_prices, price_mean, price_std = normalize_data(precios)
    
    print(f"   Mileage: mean={km_mean:.2f}, std={km_std:.2f}")
    print(f"   Prices:  mean={price_mean:.2f}, std={price_std:.2f}")
    
    # Initialize parameters
    theta0 = 0.0
    theta1 = 0.0
    m = len(kilometros)  # Number of training examples
    
    print(f"   Training with {m} data points")
    
    # Gradient descent on normalized data
    for i in range(iterations):
        # Calculate predictions for all examples (normalized)
        predictions = estimate_price_normalized(normalized_km, theta0, theta1)
        
        # Calculate errors (normalized)
        errors = predictions - normalized_prices
        
        # Calculate gradients
        gradient_theta0 = (1/m) * np.sum(errors)
        gradient_theta1 = (1/m) * np.sum(errors * normalized_km)
        
        # Update parameters
        theta0 = theta0 - learning_rate * gradient_theta0
        theta1 = theta1 - learning_rate * gradient_theta1
        
        # Print progress every 200 iterations
        if i % 200 == 0:
            cost = (1/(2*m)) * np.sum(errors**2)
            print(f"   Iteration {i:4d}: Cost = {cost:10.6f}, θ0 = {theta0:8.6f}, θ1 = {theta1:8.6f}")
    
    # Final cost calculation
    final_predictions = estimate_price_normalized(normalized_km, theta0, theta1)
    final_cost = (1/(2*m)) * np.sum((final_predictions - normalized_prices)**2)
    
    print(f"\n✅ Training completed!")
    print(f"   Final θ0 (normalized): {theta0:.6f}")
    print(f"   Final θ1 (normalized): {theta1:.6f}")
    print(f"   Final cost: {final_cost:.6f}")
    
    # Return parameters and normalization values for later use
    return theta0, theta1, km_mean, km_std, price_mean, price_std

def save_model(theta0, theta1, km_mean, km_std, price_mean, price_std, filename="trained_model.json"):
    """Save model parameters and normalization data to JSON file in the data directory"""
    import os
    
    # Ensure the data directory exists
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"📁 Created '{data_dir}' directory")
    
    # Create full path for the model file
    filepath = os.path.join(data_dir, filename)
    
    model_data = {
        "theta0": float(theta0),
        "theta1": float(theta1),
        "km_mean": float(km_mean),
        "km_std": float(km_std),
        "price_mean": float(price_mean),
        "price_std": float(price_std)
    }
    
    try:
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=4)
        print(f"💾 Model parameters saved to '{filepath}'")
        print(f"   θ0 = {theta0:.6f}")
        print(f"   θ1 = {theta1:.6f}")
        print(f"   Normalization parameters included for accurate predictions")
        return True
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return False

def main():
    print("=" * 60)
    print("║" + " " * 58 + "║")
    print("║" + " " * 12 + "🚗 LINEAR REGRESSION TRAINER 🎯" + " " * 15 + "║")
    print("║" + " " * 14 + "Training Engine for Car Prices" + " " * 14 + "║")
    print("║" + " " * 58 + "║")
    print("=" * 60)

    print("\n📄 Opening and parsing csv:")
    kilometros, precios = open_and_parse()
    if kilometros is None or precios is None:
        print("❌ Aborting training...")
        return
    print("✅ Successfully parsed csv")
    
    # Train the linear regression model with normalization
    theta0, theta1, km_mean, km_std, price_mean, price_std = train_linear_regression(
        kilometros, precios, learning_rate=0.1, iterations=1000
    )
    
    # Save model parameters including normalization data
    save_model(theta0, theta1, km_mean, km_std, price_mean, price_std, "trained_model.json")
    
    # Show the trained model information
    print(f"\n📊 Trained Model Information:")
    print(f"   Normalized θ0: {theta0:.6f}")
    print(f"   Normalized θ1: {theta1:.6f}")
    print(f"   Mileage normalization: mean={km_mean:.2f}, std={km_std:.2f}")
    print(f"   Price normalization: mean={price_mean:.2f}, std={price_std:.2f}")
    
    print(f"\n✅ Training completed successfully!")
    print(f"   All parameters saved to 'data/trained_model.json'")


if __name__== '__main__':
    main()