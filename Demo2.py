import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

datasetPath = "processedData.csv"

listCsvFiles = glob.glob(os.path.join(datasetPath, "*.csv"))

dataFrameList = [pd.read_csv(file) for file in listCsvFiles]

dataFrame = pd.concat(dataFrameList, ignore_index=True)


print(dataFrame.info())


for col in dataFrame.select_dtypes(include=["int64"]).columns:
    dataFrame[col] = dataFrame[col].astype("int32")

for col in dataFrame.select_dtypes(include=["float64"]).columns:
    dataFrame[col] = dataFrame[col].astype("float32")

for col in dataFrame.select_dtypes(include=["object"]).columns:  # Convert categorical data
    dataFrame[col] = dataFrame[col].astype("category")

#print(dataFrame.info())  # Check reduced memory usage


#print("Combined Data:\n", dataFrame.last)
def butter_lowpass(cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order)
    return filtfilt(b, a, data)

# Assuming that the dataset has numeric sensor data
numeric_cols = dataFrame.select_dtypes(include=[np.number]).columns

# Apply low-pass filter (example: cutoff frequency = 1 Hz, sampling frequency = 10 Hz)
cutoff = 1.0  # Hz
fs = 10.0  # Hz

for col in numeric_cols:
    dataFrame[col] = apply_lowpass_filter(dataFrame[col], cutoff, fs)

# Visualize the first few rows after filtering
plt.figure(figsize=(10, 6))
plt.plot(dataFrame[numeric_cols[0]], label='Filtered Data')
plt.title('Low-Pass Filtered Data (Example)')
plt.legend()
plt.show()

### Normalization ###

# 1. Min-Max Scaling
min_max_scaler = MinMaxScaler()
dataFrame[numeric_cols] = min_max_scaler.fit_transform(dataFrame[numeric_cols])

# 2. Z-score Normalization (Standardization)
z_scaler = StandardScaler()
dataFrame[numeric_cols] = z_scaler.fit_transform(dataFrame[numeric_cols])

# Visualize the normalized dataFrame
plt.figure(figsize=(10, 6))
plt.plot(dataFrame[numeric_cols[0]], label='Normalized Data')
plt.title('Z-Score Normalized Data (Example)')
plt.legend()
plt.show()

# Save the processed data
processed_file_path = r"C:\Users\Joaol\Desktop\CS156Folder\Demo\processedData.csv"
dataFrame.to_csv(processed_file_path, index=False)

print(f"Noise reduction and normalization complete. Processed dataset saved at: {processed_file_path}")
#print(dataFrame.info())
#print(dataFrame.describe())


'''plt.figure(figsize=(8, 5))
sns.scatterplot(x=dataFrame["X-accel"], y=dataFrame["Y-accel"])
plt.title("Scatter Plot of X-Accel vs Y-Accel")
plt.xlabel("X-Accel")
plt.ylabel("Y-Accel")
plt.show()
'''
def plot_histograms(data):
    """Plot histograms for all attributes except 'id' and 'timestamp'."""
    
    # Exclude 'id' and 'timestamp'
    exclude_cols = {"timestamp"}
    
    # Select numeric and categorical columns
    num_cols = [col for col in data.select_dtypes(include=["number"]).columns if col.lower() not in exclude_cols]
    cat_cols = [col for col in data.select_dtypes(include=["category", "object"]).columns if col.lower() not in exclude_cols]
    
    total_cols = len(num_cols) + len(cat_cols)  # Total attributes to plot
    plt.figure(figsize=(15, 8))  # Set figure size
    
    for i, col in enumerate(num_cols + cat_cols, 1):  # Loop through all selected columns
        plt.subplot((total_cols + 2) // 3, 3, i)  # Organize into 3 columns
        
        if col in num_cols:
            sns.histplot(data[col], bins=50, color="royalblue", edgecolor="black", alpha=0.7)
        else:
            sns.countplot(x=data[col], palette="coolwarm")  # Bar chart for categorical data
        
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.xticks(rotation=45)  # Rotate category labels if needed
    
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

    
    

def plot_scatter_matrix(data):
    """Creates multiple scatter plots to visualize relationships between numerical attributes."""
    
    num_cols = [col for col in data.select_dtypes(include=["number"]).columns if col.lower() not in {"id", "timestamp"}]
    num_attrs = len(num_cols)
    
    fig, axes = plt.subplots(num_attrs - 1, num_attrs - 1, figsize=(8, 5))  # Create subplot grid
    
    plot_index = 0  # Track which subplot we're plotting

    for i, col1 in enumerate(num_cols):
        for j, col2 in enumerate(num_cols):
            if i < j:  # Avoid duplicate scatter plots
                ax = plt.subplot(num_attrs - 1, num_attrs - 1, plot_index + 1)
                sns.scatterplot(data=data, x=col1, y=col2, ax=ax, alpha=0.5)
                ax.set_xlabel(col1)
                ax.set_ylabel(col2)
                plot_index += 1  # Move to next subplot
                print("Col 1:"+col1+" Col2:"+col2)

    plt.tight_layout()
    plt.show()

def plot_pairwise_relationships(data):
    """Create a pair plot to visualize all numerical attributes, excluding 'id' and 'timestamp'."""
    
    exclude_cols = {"id", "timestamp"}
    num_cols = [col for col in data.select_dtypes(include=["number"]).columns if col.lower() not in exclude_cols]

    sns.pairplot(data[num_cols], corner=True, plot_kws={'alpha': 0.5, 's': 10})  
    plt.suptitle("Pairwise Scatter Plots of Numerical Attributes", y=1.02, fontsize=14)  
    plt.show()
# Call the plotting functions
#plot_histograms(dataFrame)
#plot_scatter_matrix(dataFrame)
#plot_pairwise_relationships(dataFrame)