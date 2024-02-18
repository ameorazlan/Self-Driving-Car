import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def analyze(csv_path, column_name='average'):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_path)
    
    # Ensure the column exists and is numeric
    if column_name in df.columns and pd.api.types.is_numeric_dtype(df[column_name]):
        # Drop NaN values for calculation purposes
        column_data = df[column_name].dropna()
        
        # Define the conversion factor from units to meters (assuming 20 units = 0.4625 meters)
        units_to_meters = 0.4625 / 20
        
        # Calculate the number of bins needed for a 0.5 meter range based on conversion
        bin_size_in_units = int(0.5 / units_to_meters)
        
        # Calculate histogram bins for the range with bins of size calculated above
        offset_bins = range(-160, 161, bin_size_in_units)
        counts, bin_edges = np.histogram(column_data - 160, bins=offset_bins)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        # Calculate percentages
        total_count = column_data.count()
        percentages = (counts / total_count) * 100

        # Plotting the distribution
        plt.figure(figsize=(10, 6))
        plt.bar(bin_centers * units_to_meters, percentages, width=bin_size_in_units * units_to_meters * 0.9, edgecolor='black')
        
        # Generate bin labels based on offsets from 160 in meters
        bin_labels = [f"{(x * units_to_meters):.2f}-{(x + bin_size_in_units) * units_to_meters:.2f} m" 
                      for x in bin_edges[:-1]]

        plt.xlabel('Offset from Center Lane (m)')
        plt.ylabel('Percentage of Occurrences (%)')
        plt.title('Distribution of Lane Divergence')
        plt.xticks(bin_centers * units_to_meters, bin_labels, rotation=45)
        plt.xlim(-160 * units_to_meters, 160 * units_to_meters)  # Limiting x-axis to match meter units
        plt.grid(True)
        plt.tight_layout()  # Adjust layout to fit labels
        plt.show()
            
        
        # Calculate statistics
        average = column_data.mean()
        minimum = column_data.min()
        maximum = column_data.max()
        total_count = column_data.count()
        counts, bin_edges = np.histogram(column_data, bins=range(0, 321, 20))

        # Calculate percentages
        percentages = (counts / total_count) * 100

        # Display statistics
        print(f"Average: {average}")
        print(f"Min: {minimum}")
        print(f"Max: {maximum}")
        print("Percentage of occurrences in each bin:")
        for i in range(len(counts)):
            print(f"{bin_edges[i]}-{bin_edges[i+1]-1}: {percentages[i]:.2f}%")
    else:
        print(f"Column '{column_name}' does not exist or is not numeric.")

csv_path = r"Algorithm_5_stats.csv"
analyze(csv_path)
