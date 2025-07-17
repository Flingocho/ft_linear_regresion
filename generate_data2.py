import numpy as np
import csv

# Datos originales
original_data = [
    [240000, 3650],
    [139800, 3800],
    [150500, 4400],
    [185530, 4450],
    [176000, 5250],
    [114800, 5350],
    [166800, 5800],
    [89000, 5990],
    [144500, 5999],
    [84000, 6200],
    [82029, 6390],
    [63060, 6390],
    [74000, 6600],
    [97500, 6800],
    [67000, 6800],
    [76025, 6900],
    [48235, 6900],
    [93000, 6990],
    [60949, 7490],
    [65674, 7555],
    [54000, 7990],
    [68500, 7990],
    [22899, 7990],
    [61789, 8290]
]

print("📊 Analyzing original data...")
mileages = [row[0] for row in original_data]
prices = [row[1] for row in original_data]

# Calcular estadísticas básicas
mileage_mean = np.mean(mileages)
mileage_std = np.std(mileages)
price_mean = np.mean(prices)
price_std = np.std(prices)

print(f"   Original data: {len(original_data)} points")
print(f"   Mileage range: {min(mileages):,} - {max(mileages):,} km")
print(f"   Price range: ${min(prices):,} - ${max(prices):,}")
print(f"   Mileage mean: {mileage_mean:.0f} km, std: {mileage_std:.0f}")
print(f"   Price mean: ${price_mean:.0f}, std: ${price_std:.0f}")

# Calcular correlación aproximada para generar datos realistas
correlation = np.corrcoef(mileages, prices)[0,1]
print(f"   Correlation: {correlation:.3f}")

print("\n🔨 Generating 100 additional realistic data points...")

# Generar 100 puntos adicionales realistas
np.random.seed(42)  # Para reproducibilidad
additional_data = []

for i in range(100):
    # Generar kilometraje en un rango realista (0-300,000 km)
    mileage = np.random.normal(mileage_mean, mileage_std * 1.2)
    mileage = max(5000, min(300000, mileage))  # Limitar rango
    
    # Calcular precio base usando correlación aproximada
    # Fórmula simplificada: precio disminuye con kilometraje
    base_price = price_mean - (mileage - mileage_mean) * 0.02
    
    # Agregar ruido realista
    noise = np.random.normal(0, price_std * 0.4)
    price = base_price + noise
    
    # Asegurar que el precio sea realista (mínimo $2000, máximo $15000)
    price = max(2000, min(15000, price))
    
    additional_data.append([int(mileage), int(price)])

# Combinar datos originales y nuevos
all_data = original_data + additional_data

print(f"✅ Generated {len(additional_data)} additional points")
print(f"   Total dataset: {len(all_data)} points")

# Ordenar por kilometraje para mejor visualización
all_data.sort(key=lambda x: x[0])

# Escribir a data2.csv
output_file = 'data/data2.csv'
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['km', 'price'])  # Header
    writer.writerows(all_data)

print(f"💾 Data saved to '{output_file}'")

# Mostrar estadísticas del nuevo dataset
new_mileages = [row[0] for row in all_data]
new_prices = [row[1] for row in all_data]

print(f"\n📈 New dataset statistics:")
print(f"   Total points: {len(all_data)}")
print(f"   Mileage range: {min(new_mileages):,} - {max(new_mileages):,} km")
print(f"   Price range: ${min(new_prices):,} - ${max(new_prices):,}")
print(f"   New correlation: {np.corrcoef(new_mileages, new_prices)[0,1]:.3f}")
print(f"   Ready for training with more data!")
