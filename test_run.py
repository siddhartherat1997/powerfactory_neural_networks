
import os
import pandas as pd
import asyncio
import time
import numpy as np
from scipy.linalg import hankel, svd
from math import ceil
from server import myAsyncServer  # Importing the updated async server
from pencil import Pencil  # Importing the Pencil function from pencil.py
import joblib
import torch
import torch.nn as nn
# Initializing the Empty DataFrame
df = pd.DataFrame(columns=['Simulation_Time', 'Active_Power_(kW)'])

# Load the trained model
class DampingFrequencyNN(nn.Module):
    def __init__(self):
        super(DampingFrequencyNN, self).__init__()
        self.fc1 = nn.Linear(200, 1024)  # First layer with 1024 neurons
        self.fc2 = nn.Linear(1024, 512)   # Second layer with 512 neurons
        self.fc3 = nn.Linear(512, 256)    # Third layer with 256 neurons
        self.fc4 = nn.Linear(256, 2)       # Output layer with 2 neurons for frequency and damping
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Initialize the model and load the state dict
model = DampingFrequencyNN()
model.load_state_dict(torch.load(r".\Trained_Files\damping_frequency_model.pth")) # CHANGE THE DIRECTORY#
model.eval()  # Set the model to evaluation mode

# Load the scaler
scaler_path = (r".\Trained_Files\scaler.pkl") # CHANGE THE DIRECTORY#
scaler = joblib.load(scaler_path)

# Function to Append Data
def append_data(timestamp, active_power):
    global df

    # Check if timestamp already exists in the DataFrame and if active power is not zero
    if timestamp not in df['Simulation_Time'].values and active_power != 0:
        # Create a new DataFrame for the new row
        new_row = pd.DataFrame({'Simulation_Time': [timestamp], 'Active_Power_(kW)': [active_power]})
        if not new_row.isna().all().all():
            # Concatenate the new row with the existing DataFrame
            df = pd.concat([df, new_row], ignore_index=True)
            print(f"Data appended: Simulation Time: {timestamp}, Active Power: {active_power}")
    else:
        print(f"Duplicate or zero power value detected at Simulation Time: {timestamp}")

# Function to Reset Buffer Data
def reset_buffer_data():
    global df
    if not df.empty:
        print(f'Buffer Reset. Total Collected: {len(df)} rows')
        df = df.iloc[0:0]  # Reset DataFrame after processing

# Main async function
async def main():
    print("Starting the server...")
    server = myAsyncServer()
    await server.start_server()

    previous_kq = None
    previous_active_power = None
    last_power_check_time = None  # Use simulation time for the power check interval
    kq_changed = False
    collecting_data = False
    collected_count = 0  # Count of collected values
    damping_values = []  # Initialize damping values list for plotting
    frequency_values = []
    initial_zeta = 40  # Start with an initial damping percentage higher than 7.5
    kq_set = False  # Flag to ensure kq is set once at -10 seconds
    recording_started = False  # Flag to start recording at -15 seconds
    last_kq_update_time = None  # Track when kq was last updated
    power_violation_detected = False  # Flag for power condition violation

    delta_kq = 0.005  # Step size for kq adjustment
    power_threshold = 0.5  # Threshold for change in active power

    while True:
        conv1_active_power = await server.get_conv1_active_power()
        simulation_time = await server.get_simulation_time()
        conv_1_voltage = await server.get_conv1_voltage_meas()
        current_kq = await server.get_conv1_kq_set()

        rounded_simulation_time = round(simulation_time, 2)
        rounded_active_power = round(conv1_active_power, 5)

        # Start recording at t = -15 seconds
        if rounded_simulation_time == -15 and not recording_started:
            print(f"Starting recording at {rounded_simulation_time} seconds.")
            recording_started = True

        # If recording has started, continue collecting data
        if recording_started and rounded_simulation_time >= -15:
            # Set kq to 0.025 when simulation time reaches exactly -10 seconds
            if rounded_simulation_time == -10 and not kq_set:
                await server.set_conv1_kq_set_1(0.025)
                print(f"kq set to 0.025 at {rounded_simulation_time} seconds.")
                kq_set = True  # Set the flag to prevent resetting the value

            # Detect if active power differs significantly
            if previous_active_power is not None and abs(rounded_active_power - previous_active_power) > power_threshold:
                print(f"Significant change in active power detected: {rounded_active_power}")
                reset_buffer_data()

            previous_active_power = rounded_active_power  # Update previous active power

            # Collect 200 unique values
            if len(df) < 200:
                append_data(rounded_simulation_time, rounded_active_power)

                if len(df) >= 200:
                    print("200 unique values collected. Using neural network for prediction...")

                    # Prepare 1x200 array for StandardScaler
                    active_power_data = df['Active_Power_(kW)'].values.reshape(1, -1)  # Shape (1, 200)

                    # Scale the active power data using the loaded StandardScaler
                    scaled_data = scaler.transform(active_power_data)

                    # Convert scaled data to PyTorch tensor
                    input_tensor = torch.FloatTensor(scaled_data)  # Shape is (1, 200)
                    
                    # Print tensor for verification
                    print(f"Scaled Tensor Data Shape: {input_tensor.shape}")

                    # Make predictions using the model
                    with torch.no_grad():
                        prediction = model(input_tensor)

                    # Extract predicted frequency and damping
                    predicted_frequency = prediction[0, 0].item()  # Get the first output
                    predicted_damping = prediction[0, 1].item()    # Get the second output

                    print(f"Predicted Frequency: {predicted_frequency:.2f} Hz")
                    print(f"Predicted Damping: {predicted_damping:.2f} %")

                    # Logic for adjusting kq based on the predicted damping and voltage
                    new_kq = current_kq  # Start with the current kq

                    if predicted_damping > 8.05:
                        voltage_deviation = abs(conv_1_voltage - 19.82)  # Check deviation from desired voltage
                        if voltage_deviation < 0.5:
                            new_kq += delta_kq  # Increase kq if within range
                        elif voltage_deviation > 0.7:
                            new_kq -= delta_kq  # Decrease kq if out of range
                    elif predicted_damping < 7.9:
                        new_kq -= delta_kq  # Decrease kq based on damping prediction

                    await server.set_conv1_kq_set_1(new_kq)  # Send the new kq to the server
                    print(f"New kq sent to PowerFactory: {new_kq}")

                    # Reset the buffer after adjusting kq
                    reset_buffer_data()

        await asyncio.sleep(0.001)

# Starting the async loop when the script is run as a standalone program
if __name__ == "__main__":
    asyncio.run(main())






