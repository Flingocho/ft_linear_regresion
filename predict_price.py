import json
import os
import numpy as np
import matplotlib.pyplot as plt

def load_model(filename="data/trained_model.json"):
    """
    Load trained model parameters from JSON file
    Returns default values (0, 0) if any error occurs
    """
    try:
        # Check if file exists
        if not os.path.exists(filename):
            print(f"❌ Model file '{filename}' not found. Using default values.")
            return 0, 0, 0, 1, 0, 1
        
        # Try to load and parse JSON
        with open(filename, 'r') as f:
            model_data = json.load(f)
        
        # Validate that all required keys exist
        required_keys = ["theta0", "theta1", "mileage_mean", "mileage_std", "price_mean", "price_std"]
        for key in required_keys:
            if key not in model_data:
                print(f"❌ Missing key '{key}' in model file. Using default values.")
                return 0, 0, 0, 1, 0, 1
        
        # Extract parameters
        theta0 = float(model_data["theta0"])
        theta1 = float(model_data["theta1"])
        mileage_mean = float(model_data["mileage_mean"])
        mileage_std = float(model_data["mileage_std"])
        price_mean = float(model_data["price_mean"])
        price_std = float(model_data["price_std"])
        
        print(f"✅ Model loaded successfully from '{filename}'")
        print(f"   θ0: {theta0:.6f}")
        print(f"   θ1: {theta1:.6f}")
        print(f"   Normalization parameters loaded")
        
        return theta0, theta1, mileage_mean, mileage_std, price_mean, price_std
        
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON format in '{filename}'. Using default values.")
        return 0, 0, 0, 1, 0, 1
    except ValueError as e:
        print(f"❌ Invalid data types in model file: {e}. Using default values.")
        return 0, 0, 0, 1, 0, 1
    except Exception as e:
        print(f"❌ Error loading model: {e}. Using default values.")
        return 0, 0, 0, 1, 0, 1

def normalize_mileage(mileage, mileage_mean, mileage_std):
    """Normalize mileage using saved parameters"""
    if mileage_std == 0:
        return 0
    return (mileage - mileage_mean) / mileage_std

def denormalize_price(normalized_price, price_mean, price_std):
    """Convert normalized price back to original scale"""
    return normalized_price * price_std + price_mean

def predict_price(mileage, theta0, theta1, mileage_mean, mileage_std, price_mean, price_std):
    """
    Predict car price based on mileage using trained model
    Returns predicted price in original scale
    """
    # Normalize input mileage
    normalized_mileage = normalize_mileage(mileage, mileage_mean, mileage_std)
    
    # Apply linear regression model (normalized)
    normalized_price = theta0 + theta1 * normalized_mileage
    
    # Denormalize to get actual price
    actual_price = denormalize_price(normalized_price, price_mean, price_std)
    
    return actual_price

def estimate_price_normalized(normalized_mileage, theta0, theta1):
    """Calculate estimated normalized price using linear regression formula"""
    return theta0 + theta1 * normalized_mileage

def load_training_data(filename="data/data.csv"):
    """Load original training data for visualization"""
    try:
        with open(filename, "r") as file:
            content = file.read()
        
        lines = content.strip().split('\n')
        if not lines or lines[0].strip() != "km,price":
            return None, None
        
        mileage = []
        prices = []
        
        for i in range(1, len(lines)):
            row = lines[i].split(',')
            if len(row) >= 2:
                try:
                    mileage.append(float(row[0]))
                    prices.append(float(row[1]))
                except ValueError:
                    continue
        
        return mileage, prices
    except:
        return None, None

def visualize_prediction(mileage, prices, theta0, theta1, mileage_mean, mileage_std, price_mean, price_std, pred_mileage, pred_price):
    """
    Visualize training data with regression line and highlight the new prediction
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
        
        # Create gradient scatter plot for training data (smaller points)
        scatter = ax.scatter(mileage_array, price_array, 
                           c=price_array, cmap='plasma', 
                           s=80, alpha=0.6,  # Smaller and more transparent
                           edgecolors='white', linewidths=1,
                           label='Training Data')
        
        # Add the prediction point (larger and highlighted)
        pred_scatter = ax.scatter([pred_mileage], [pred_price], 
                                c='#FF6B9D', s=300, alpha=0.9,
                                edgecolors='white', linewidths=3,
                                marker='*', 
                                label=f'Prediction: {pred_mileage:,.0f} km → ${pred_price:,.2f}',
                                zorder=5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=30)
        cbar.set_label('Price ($)', fontsize=12, color='white', fontweight='bold')
        cbar.ax.tick_params(colors='white', labelsize=10)
        
        # Generate points for the regression line (extend to virtually infinite range)
        mileage_min, mileage_max = mileage_array.min(), mileage_array.max()
        # Include prediction point in range calculation
        full_mileage_min = min(mileage_min, pred_mileage)
        full_mileage_max = max(mileage_max, pred_mileage)
        data_range = full_mileage_max - full_mileage_min
        
        # Extend the line far beyond the data points
        extension = max(data_range * 5, 100000)
        mileage_range = np.linspace(full_mileage_min - extension, full_mileage_max + extension, 500)
        
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
        
        # Create annotation for mouse hover on training data points
        point_annot = ax.annotate('', xy=(0,0), xytext=(20,20), textcoords="offset points",
                           bbox=dict(boxstyle="round,pad=0.5", facecolor='#21262D', 
                                   edgecolor='#FF6B9D', alpha=0.9),
                           arrowprops=dict(arrowstyle="->", color='#FF6B9D'),
                           fontsize=10, color='white', fontweight='bold')
        point_annot.set_visible(False)
        
        # Create annotation for prediction point
        pred_annot = ax.annotate('', xy=(0,0), xytext=(20,20), textcoords="offset points",
                           bbox=dict(boxstyle="round,pad=0.5", facecolor='#21262D', 
                                   edgecolor='#FF6B9D', alpha=0.9),
                           arrowprops=dict(arrowstyle="->", color='#FF6B9D'),
                           fontsize=10, color='white', fontweight='bold')
        pred_annot.set_visible(False)
        
        def estimate_price_at_mileage(km):
            """Calculate estimated price for given mileage"""
            km_normalized = (km - mileage_mean) / mileage_std
            price_normalized = estimate_price_normalized(km_normalized, theta0, theta1)
            return denormalize_price(price_normalized, price_mean, price_std)
        
        def find_closest_point(event):
            """Find the closest data point to mouse position in display (pixel) coordinates"""
            # Transform training data points to display coordinates
            xy_disp = ax.transData.transform(np.column_stack([mileage_array, price_array]))
            mouse_disp = np.array([event.x, event.y])
            distances = np.linalg.norm(xy_disp - mouse_disp, axis=1)
            closest_idx = np.argmin(distances)
            closest_distance = distances[closest_idx]
            # Marker size in points^2, convert to pixels
            marker_size = scatter.get_sizes()[0] if scatter.get_sizes().size > 0 else 80
            marker_radius = np.sqrt(marker_size) / 2.0
            marker_radius_pixels = marker_radius * fig.dpi / 72.0
            if closest_distance <= marker_radius_pixels:
                return closest_idx, mileage_array[closest_idx], price_array[closest_idx]
            return None, None, None
        
        def find_prediction_point(event):
            """Check if mouse is over prediction point"""
            pred_disp = ax.transData.transform([[pred_mileage, pred_price]])[0]
            mouse_disp = np.array([event.x, event.y])
            distance = np.linalg.norm(mouse_disp - pred_disp)
            # Prediction marker is larger
            marker_radius_pixels = np.sqrt(300) / 2.0 * fig.dpi / 72.0
            if distance <= marker_radius_pixels:
                return True
            return False
        
        def on_hover(event):
            """Handle mouse hover events with strict, pixel-perfect detection"""
            if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
                # 1. Check for prediction point hover
                if find_prediction_point(event):
                    pred_annot.xy = (pred_mileage, pred_price)
                    text = f'NEW PREDICTION\nMileage: {pred_mileage:,.0f} km\nPredicted Price: ${pred_price:,.2f}'
                    pred_annot.set_text(text)
                    pred_annot.set_visible(True)
                    point_annot.set_visible(False)
                    line_annot.set_visible(False)
                    fig.canvas.draw_idle()
                    return
                
                # 2. Check for training data point hover
                point_idx, point_km, point_price = find_closest_point(event)
                if point_idx is not None:
                    point_annot.xy = (point_km, point_price)
                    text = f'Training Data #{point_idx + 1}\nMileage: {point_km:,.0f} km\nActual Price: ${point_price:,.2f}'
                    point_annot.set_text(text)
                    point_annot.set_visible(True)
                    pred_annot.set_visible(False)
                    line_annot.set_visible(False)
                    fig.canvas.draw_idle()
                    return
                
                # 3. Check for regression line hover
                mouse_km = event.xdata
                mouse_price = event.ydata
                estimated_price = estimate_price_at_mileage(mouse_km)
                mouse_disp = np.array([event.x, event.y])
                line_disp = ax.transData.transform([[mouse_km, estimated_price]])[0]
                distance_to_line = np.linalg.norm(mouse_disp - line_disp)
                if distance_to_line <= 2.0:
                    line_annot.xy = (mouse_km, estimated_price)
                    text = f'Regression Line\nMileage: {mouse_km:,.0f} km\nEstimated Price: ${estimated_price:,.2f}'
                    line_annot.set_text(text)
                    line_annot.set_visible(True)
                    point_annot.set_visible(False)
                    pred_annot.set_visible(False)
                    fig.canvas.draw_idle()
                    return
                
                # 4. Hide all tooltips
                line_annot.set_visible(False)
                point_annot.set_visible(False)
                pred_annot.set_visible(False)
                fig.canvas.draw_idle()
            else:
                line_annot.set_visible(False)
                point_annot.set_visible(False)
                pred_annot.set_visible(False)
                fig.canvas.draw_idle()
        
        # Connect the hover event
        fig.canvas.mpl_connect('motion_notify_event', on_hover)
        
        # Plot customization
        ax.set_title('Car Price Prediction - Model Visualization', 
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
        
        # Set reasonable axis limits centered on the data including prediction
        all_mileage = list(mileage) + [pred_mileage]
        all_prices = list(prices) + [pred_price]
        data_range_x = max(all_mileage) - min(all_mileage)
        data_range_y = max(all_prices) - min(all_prices)
        
        # Add some margin around the data (30% on each side)
        margin_x = max(data_range_x * 0.3, 10000)  # At least 10k margin
        margin_y = max(data_range_y * 0.3, 1000)   # At least 1k margin
        
        ax.set_xlim(min(all_mileage) - margin_x, max(all_mileage) + margin_x)
        ax.set_ylim(min(all_prices) - margin_y, max(all_prices) + margin_y)
        
        # Legend
        legend = ax.legend(fontsize=12, frameon=True, fancybox=True, 
                          shadow=True, facecolor='#21262D', edgecolor='#30363D')
        legend.get_frame().set_alpha(0.9)
        for text in legend.get_texts():
            text.set_color('white')
        
        # Adjust layout to prevent clipping
        plt.tight_layout()
        plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.1)
        plt.show()
        
    except ImportError:
        print("❌ matplotlib not installed.")
        print("   To visualize predictions, install it with:")
        print("   pip install matplotlib")
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")

def main():
    print("=" * 60)
    print("║" + " " * 58 + "║")
    print("║" + " " * 16 + "🚗 CAR PRICE PREDICTOR 💰" + " " * 17 + "║")
    print("║" + " " * 12 + "Predict prices using trained model" + " " * 12 + "║")
    print("║" + " " * 58 + "║")
    print("=" * 60)
    
    print("\n📁 Loading trained model...")
    # Load model parameters (defaults to 0,0 if any error)
    theta0, theta1, mileage_mean, mileage_std, price_mean, price_std = load_model()
    
    # Check if using default values
    if theta0 == 0 and theta1 == 0:
        print("\n⚠️  WARNING: Using default values (0, 0)")
        print("   Please train a model first using train_model.py")
        print("   Predictions will not be accurate!\n")
    
    print("\n🔢 Car Price Calculator")
    print("   Enter mileage to get price prediction")
    print("   Type 'quit' or 'exit' to stop\n")
    
    while True:
        try:
            # Get user input
            user_input = input("🚗 Enter car mileage (km): ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Exiting... Bye!")
                break
            
            # Try to convert to float
            mileage = float(user_input)
            
            # Validate input
            if mileage < 0:
                print("❌ Mileage cannot be negative. Please enter a valid value.")
                continue
            
            # Make prediction
            predicted_price = predict_price(mileage, theta0, theta1, mileage_mean, mileage_std, price_mean, price_std)
            
            # Display result
            print(f"💰 Predicted price: ${predicted_price:,.2f}")
            
            # Show model equation for reference (if not default values)
            if not (theta0 == 0 and theta1 == 0):
                print(f"   Model: price = {theta0:.3f} + {theta1:.3f} × normalized_mileage")
            
            # Ask if user wants to visualize the prediction
            if not (theta0 == 0 and theta1 == 0):  # Only offer visualization if we have a trained model
                print(f"\n📊 Do you want to see this prediction on the graph? (y/n): ", end="")
                viz_choice = input().strip().lower()
                if viz_choice in ['y', 'yes']:
                    print("📈 Loading training data for visualization...")
                    training_mileage, training_prices = load_training_data()
                    if training_mileage and training_prices:
                        print("🎨 Creating visualization...")
                        visualize_prediction(training_mileage, training_prices, theta0, theta1, 
                                           mileage_mean, mileage_std, price_mean, price_std, 
                                           mileage, predicted_price)
                    else:
                        print("❌ Could not load training data for visualization.")
            
            print()  # Empty line for readability
            
        except ValueError:
            print("❌ Invalid input. Please enter a numeric value for mileage.")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")

if __name__ == '__main__':
    main()
