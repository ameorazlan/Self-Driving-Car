import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def analyze(csv_path, column_name='average'):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_path)
    
    if column_name in df.columns and pd.api.types.is_numeric_dtype(df[column_name]):
        # Drop NaN values for calculation purposes
        column_data = df[column_name].dropna()
        
        # Subtract 160 from each value in the column
        adjusted_column_data = (column_data - 160) / 80
        
        # Plotting the histogram of the adjusted data
        plt.hist(adjusted_column_data, bins=32, range=(-2, 2), edgecolor='black')
        plt.title(f'Federated Client 3 (Stage 1) Deviation From Center Lane')
        plt.xlabel(f'Meters')
        plt.ylabel('Frequency')
        plt.grid(True)
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
        print(f"Column '{column_name}' does notw exist or is not numeric.")

csv_path = r"PyTorch_Federated_Client_3.csv"
analyze(csv_path)
