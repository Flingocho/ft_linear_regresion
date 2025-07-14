import numpy as np
import json
import matplotlib.pyplot as plt

def open_and_parse():
    """Open the file in data/data.csv and parse the content"""
    # Open the file with read permission and save it in file
    with open("data/data.csv", "r") as file:
        content = file.read()
    
    # Split by lines first, then by commas
    lines = content.strip().split('\n')  # Split into lines
    
    # Separate lists for mileage and prices
    mileage = []
    prices = []
    
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
        
        # Save mileage and prices in separate lists
        if len(row) >= 2:
            try:
                mileage.append(float(row[0]))  # Convert to number
                prices.append(float(row[1]))     # Convert to number
            except ValueError:
                print(f"❌ Error converting line {i+1}: {row}")
                return None, None
            
            if mileage and prices and (mileage[-1] < 0 or prices[-1] < 0):
                # Line number to show to user (i+1 because we always have header)
                print(f"❌ Error: negative value in row: {i+1}")
                print(f"      -> {lines[i]}")
                return None, None

    print(f"🧮 Total training points: {len(mileage)} points")
    
    return mileage, prices  # Return both lists separately


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

def train_linear_regression(mileage, prices, learning_rate=0.01, iterations=1000):
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
    normalized_mileage, mileage_mean, mileage_std = normalize_data(mileage)
    normalized_prices, price_mean, price_std = normalize_data(prices)
    
    print(f"   Mileage: mean={mileage_mean:.2f}, std={mileage_std:.2f}")
    print(f"   Prices:  mean={price_mean:.2f}, std={price_std:.2f}")
    
    # Initialize parameters
    theta0 = 0.0
    theta1 = 0.0
    m = len(mileage)  # Number of training examples
    
    print(f"   Training with {m} data points")
    
    # Gradient descent on normalized data
    for i in range(iterations):
        # Calculate predictions for all examples (normalized)
        predictions = estimate_price_normalized(normalized_mileage, theta0, theta1)
        
        # Calculate errors (normalized)
        errors = predictions - normalized_prices
        
        # Calculate gradients
        gradient_theta0 = (1/m) * np.sum(errors)
        gradient_theta1 = (1/m) * np.sum(errors * normalized_mileage)
        
        # Update parameters
        theta0 = theta0 - learning_rate * gradient_theta0
        theta1 = theta1 - learning_rate * gradient_theta1
        
        # Print progress every 200 iterations
        if i % 200 == 0:
            cost = (1/(2*m)) * np.sum(errors**2)
            print(f"   Iteration {i:4d}: Cost = {cost:10.6f}, θ0 = {theta0:8.6f}, θ1 = {theta1:8.6f}")
    
    # Final cost calculation
    final_predictions = estimate_price_normalized(normalized_mileage, theta0, theta1)
    final_cost = (1/(2*m)) * np.sum((final_predictions - normalized_prices)**2)
    
    print(f"\n✅ Training completed!")
    print(f"   Final θ0 (normalized): {theta0:.6f}")
    print(f"   Final θ1 (normalized): {theta1:.6f}")
    print(f"   Final cost: {final_cost:.6f}")
    
    # Return parameters and normalization values for later use
    return theta0, theta1, mileage_mean, mileage_std, price_mean, price_std

def save_model(theta0, theta1, mileage_mean, mileage_std, price_mean, price_std, filename="trained_model.json"):
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
        "mileage_mean": float(mileage_mean),
        "mileage_std": float(mileage_std),
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


def visualize_regression(mileage, prices, theta0, theta1, mileage_mean, mileage_std, price_mean, price_std):
    """
    Visualize the original data points and the fitted regression line with interactive hover on both points and line
    """    
    try:
        # Set dark theme
        plt.style.use('dark_background')
        
        # Create figure with custom size and styling
        fig, ax = plt.subplots(figsize=(14, 9), facecolor='#0D1117')
        ax.set_facecolor('#161B22')
        
        # Convert to numpy arrays
        mileage_array = np.array(mileage)
        price_array = np.array(prices)
        
        # Create gradient scatter plot
        scatter = ax.scatter(mileage_array, price_array, 
                           c=price_array, cmap='plasma', 
                           s=120, alpha=0.8, 
                           edgecolors='white', linewidths=1.5,
                           label='Training Data')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=30)
        cbar.set_label('Price ($)', fontsize=12, color='white', fontweight='bold')
        cbar.ax.tick_params(colors='white', labelsize=10)
        
        # Generate points for the regression line (extend to virtually infinite range)
        mileage_min, mileage_max = mileage_array.min(), mileage_array.max()
        data_range = mileage_max - mileage_min
        # Extend the line far beyond the data points (5x the data range on each side)
        extension = max(data_range * 5, 100000)  # At least 100km extension
        mileage_range = np.linspace(mileage_min - extension, mileage_max + extension, 500)
        
        # Normalize the range, apply model, then denormalize
        mileage_range_normalized = (mileage_range - mileage_mean) / mileage_std
        price_range_normalized = estimate_price_normalized(mileage_range_normalized, theta0, theta1)
        price_range = denormalize_price(price_range_normalized, price_mean, price_std)
        
        # Plot regression line
        line, = ax.plot(mileage_range, price_range, color='#00D4FF', linewidth=2, 
                       label=f'Regression Line: θ₀={theta0:.3f}, θ₁={theta1:.3f}',
                       alpha=0.9, linestyle='-')
        
        # Create annotation for mouse hover on regression line
        line_annot = ax.annotate('', xy=(0,0), xytext=(20,20), textcoords="offset points",
                           bbox=dict(boxstyle="round,pad=0.5", facecolor='#21262D', 
                                   edgecolor='#00D4FF', alpha=0.9),
                           arrowprops=dict(arrowstyle="->", color='#00D4FF'),
                           fontsize=10, color='white', fontweight='bold')
        line_annot.set_visible(False)
        
        # Create annotation for mouse hover on data points
        point_annot = ax.annotate('', xy=(0,0), xytext=(20,20), textcoords="offset points",
                           bbox=dict(boxstyle="round,pad=0.5", facecolor='#21262D', 
                                   edgecolor='#FF6B9D', alpha=0.9),
                           arrowprops=dict(arrowstyle="->", color='#FF6B9D'),
                           fontsize=10, color='white', fontweight='bold')
        point_annot.set_visible(False)
        
        def estimate_price_at_mileage(km):
            """Calculate estimated price for given mileage"""
            km_normalized = (km - mileage_mean) / mileage_std
            price_normalized = estimate_price_normalized(km_normalized, theta0, theta1)
            return denormalize_price(price_normalized, price_mean, price_std)
        
        def find_closest_point(event):
            """Find the closest data point to mouse position in display (pixel) coordinates"""
            # Transform data points to display coordinates
            xy_disp = ax.transData.transform(np.column_stack([mileage_array, price_array]))
            mouse_disp = np.array([event.x, event.y])
            distances = np.linalg.norm(xy_disp - mouse_disp, axis=1)
            closest_idx = np.argmin(distances)
            closest_distance = distances[closest_idx]
            # Marker size in points^2, convert to pixels (approximate)
            marker_size = scatter.get_sizes()[0] if scatter.get_sizes().size > 0 else 120
            marker_radius = np.sqrt(marker_size) / 2.0
            # Convert marker radius from points to pixels
            # 1 point = 1/72 inch, dpi = fig.dpi
            marker_radius_pixels = marker_radius * fig.dpi / 72.0
            # Use a strict threshold: only trigger if mouse is within marker pixel radius
            if closest_distance <= marker_radius_pixels:
                return closest_idx, mileage_array[closest_idx], price_array[closest_idx]
            return None, None, None
        
        def on_hover(event):
            """Handle mouse hover events with strict, pixel-perfect detection"""
            if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
                # 1. Check for data point hover (pixel-perfect)
                point_idx, point_km, point_price = find_closest_point(event)
                if point_idx is not None:
                    # Show data point tooltip only
                    point_annot.xy = (point_km, point_price)
                    text = f'Data Point #{point_idx + 1}\nMileage: {point_km:,.0f} km\nActual Price: ${point_price:,.2f}'
                    point_annot.set_text(text)
                    point_annot.set_visible(True)
                    line_annot.set_visible(False)
                    fig.canvas.draw_idle()
                    return
                # 2. Check for regression line hover (pixel-perfect, only if not on a point)
                # Find closest x on the regression line to mouse x
                mouse_km = event.xdata
                mouse_price = event.ydata
                estimated_price = estimate_price_at_mileage(mouse_km)
                # Convert (mouse_km, mouse_price) and (mouse_km, estimated_price) to display coords
                mouse_disp = np.array([event.x, event.y])
                line_disp = ax.transData.transform([[mouse_km, estimated_price]])[0]
                distance_to_line = np.linalg.norm(mouse_disp - line_disp)
                # Use a strict threshold: 2 pixels
                if distance_to_line <= 2.0:
                    line_annot.xy = (mouse_km, estimated_price)
                    text = f'Regression Line\nMileage: {mouse_km:,.0f} km\nEstimated Price: ${estimated_price:,.2f}'
                    line_annot.set_text(text)
                    line_annot.set_visible(True)
                    point_annot.set_visible(False)
                    fig.canvas.draw_idle()
                    return
                # 3. Hide both if not exactly on a point or the line
                line_annot.set_visible(False)
                point_annot.set_visible(False)
                fig.canvas.draw_idle()
            else:
                line_annot.set_visible(False)
                point_annot.set_visible(False)
                fig.canvas.draw_idle()
        
        # Connect the hover event
        fig.canvas.mpl_connect('motion_notify_event', on_hover)
        
        # Plot customization
        ax.set_title('Car Price Prediction using Linear Regression', 
                    fontsize=18, fontweight='bold', color='white', pad=25)
        ax.set_xlabel('Mileage (km)', fontsize=14, fontweight='bold', color='white')
        ax.set_ylabel('Price ($)', fontsize=14, fontweight='bold', color='white')
        
        # Grid
        ax.grid(True, alpha=0.2, linestyle='--', color='#30363D', linewidth=0.8)
        
        # Format axes
        ax.tick_params(colors='white', labelsize=11)
        ax.spines['bottom'].set_color('#30363D')
        ax.spines['top'].set_color('#30363D')
        ax.spines['right'].set_color('#30363D')
        ax.spines['left'].set_color('#30363D')
        
        # Format axes numbers
        ax.ticklabel_format(style='plain', axis='both')
        
        # Set reasonable axis limits centered on the data
        data_range_x = mileage_max - mileage_min
        data_range_y = price_array.max() - price_array.min()
        
        # Add some margin around the data (30% on each side)
        margin_x = data_range_x * 0.3
        margin_y = data_range_y * 0.3
        
        ax.set_xlim(mileage_min - margin_x, mileage_max + margin_x)
        ax.set_ylim(price_array.min() - margin_y, price_array.max() + margin_y)
        
        # Legend
        legend = ax.legend(fontsize=12, frameon=True, fancybox=True, 
                          shadow=True, facecolor='#21262D', edgecolor='#30363D')
        legend.get_frame().set_alpha(0.9)
        for text in legend.get_texts():
            text.set_color('white')
        
        # Adjust layout to prevent clipping
        plt.tight_layout()

        # Add subtle animation effect
        plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.1)
        plt.show()
        
    except ImportError:
        print("❌ matplotlib not installed.")
        print("   To visualize data with dark theme, install it with:")
        print("   pip install matplotlib")
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")

def main():
    print("=" * 60)
    print("║" + " " * 58 + "║")
    print("║" + " " * 14 + "🚗 LINEAR REGRESSION TRAINER 🎯" + " " * 13 + "║")
    print("║" + " " * 14 + "Training Engine for Car Prices" + " " * 14 + "║")
    print("║" + " " * 58 + "║")
    print("=" * 60)

    print("\n📄 Opening and parsing csv:")
    mileage, prices = open_and_parse()
    if mileage is None or prices is None:
        print("❌ Aborting training...")
        return
    print("✅ Successfully parsed csv")
    
    # Train the linear regression model with normalization
    theta0, theta1, mileage_mean, mileage_std, price_mean, price_std = train_linear_regression(
        mileage, prices, learning_rate=0.1, iterations=1000
    )
    
    # Save model parameters including normalization data
    save_model(theta0, theta1, mileage_mean, mileage_std, price_mean, price_std, "trained_model.json")
    
    # Show the trained model information
    print(f"\n📊 Trained Model Information:")
    print(f"   Normalized θ0: {theta0:.6f}")
    print(f"   Normalized θ1: {theta1:.6f}")
    print(f"   Mileage normalization: mean={mileage_mean:.2f}, std={mileage_std:.2f}")
    print(f"   Price normalization: mean={price_mean:.2f}, std={price_std:.2f}")
    
    print(f"\n✅ Training completed successfully!")
    print(f"   All parameters saved to 'data/trained_model.json'")
    
    # Optional visualization
    print(f"\n🎨 Do you want to visualize the data and regression line? (y/n)")
    choice = input().lower()
    if choice == 'y' or choice == 'yes':
        visualize_regression(mileage, prices, theta0, theta1, mileage_mean, mileage_std, price_mean, price_std)
    print(f"Exiting... ¡Bye!")

if __name__ == '__main__':
    main()