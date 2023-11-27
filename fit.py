from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# Load and preprocess the dataset
dataset_fp = Path('./data/covid_19_indonesia.csv')
dataset = pd.read_csv(dataset_fp)
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset = dataset[dataset['Location'] == 'Indonesia']

# %%
# Parameters for window size and step size
window_size = 240  # days
step_size = 1  # days


# Function for the custom model
def poly_logistic_model(x, a, b, c, d, e, f, g, h):
    return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e + f / (1 + np.exp(-g * (x - h)))


# Initial guesses for the parameters
initial_guesses = [0.1] * 8

# Setting up the animation
fig, ax = plt.subplots(figsize=(10, 6))
plt.title('COVID-19 New Cases in Indonesia')
plt.xlabel('Date')
plt.ylabel('New Cases')


def animate(i):
    start_date = dataset['Date'].min() + pd.Timedelta(days=i * step_size)
    end_date = start_date + pd.Timedelta(days=window_size)

    # Trim dataset based on the window
    windowed_dataset = dataset[(dataset['Date'] >= start_date) & (dataset['Date'] <= end_date)]

    # Extracting date and new cases values
    x = windowed_dataset['Date']
    y = windowed_dataset['New Cases']
    x_days = (x - dataset['Date'].min()) / np.timedelta64(1, 'D')
    x_days_normalized = x_days - x_days.min()

    # Fit the model to the data
    params, _ = curve_fit(poly_logistic_model, x_days_normalized.values, y, p0=initial_guesses)

    # Predictions using the fitted model
    y_pred = poly_logistic_model(x_days_normalized.values, *params)

    # Plotting
    ax.clear()
    ax.scatter(x, y, label='Actual Data')
    ax.plot(x, y_pred, label='4th Degree Poly + Logistic Model', color='blue')
    ax.legend()


# Creating the animation
ani = animation.FuncAnimation(fig, animate,
                              frames=np.arange(0, (dataset['Date'].max() - dataset['Date'].min()).days // step_size),
                              repeat=False)


def print_progress(current_frame, total_frames):
    progress = (current_frame / total_frames) * 100
    print(f"Saving progress: {progress:.2f}%")


# Save the animation
ani.save('new_cases_indonesia_fast.gif', fps=240, writer='imagemagick', progress_callback=print_progress)

plt.show()
