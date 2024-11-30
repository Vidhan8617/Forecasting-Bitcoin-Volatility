import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
from scipy.interpolate import make_interp_spline
import numpy as np


# Load the data
df = pd.read_csv('BIT-USD.csv')


# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Set 'Date' as the index
df.set_index('Date', inplace=True)

# Feature Engineering: Create additional time-based features
df['Year'] = df.index.year
df['Month'] = df.index.month
df['Day'] = df.index.day
df['DayOfWeek'] = df.index.dayofweek

# Define features and target
X = df[['Open', 'High', 'Low', 'Volume', 'Year', 'Month', 'Day', 'DayOfWeek']]
y = df['Close']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = r2 * 100

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')
print(f'Accuracy: {accuracy}%')

# Plot the actual vs predicted values
plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, y_pred, label='Predicted')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Actual vs Predicted Close Price')
plt.legend()
plt.show()

# Group by year and calculate the average Close price
avg_close_per_year = df.groupby('Year')['Close'].mean()

# Find the year with the highest and lowest average close price
max_year = avg_close_per_year.idxmax()
min_year = avg_close_per_year.idxmin()

# Create a smoother curve using B-spline interpolation
years = avg_close_per_year.index
values = avg_close_per_year.values

# Generate a denser x-axis for smooth plotting
years_smooth = np.linspace(years.min(), years.max(), 500)
spline = make_interp_spline(years, values, k=3)  # B-spline of degree 3
values_smooth = spline(years_smooth)

# Plot the smooth average Close price
plt.figure(figsize=(12, 6))
plt.plot(years_smooth, values_smooth, color='orange', label='Average Close Price' , linewidth=2)

# Annotate the highest point
plt.annotate(f'Highest: {avg_close_per_year[max_year]:.2f}',
             xy=(max_year, avg_close_per_year[max_year]),
             xytext=(max_year, avg_close_per_year[max_year] + 1000),
             arrowprops=dict(facecolor='green', arrowstyle='->'),
             fontsize=10)

# Annotate the lowest point
plt.annotate(f'Lowest: {avg_close_per_year[min_year]:.2f}',
             xy=(min_year, avg_close_per_year[min_year]),
             xytext=(min_year, avg_close_per_year[min_year] - 1000),
             arrowprops=dict(facecolor='red', arrowstyle='->'),
             fontsize=10)

# Add plot details
plt.title('Average Closing Price Per Year  ', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Average Closing Price (USD)', fontsize=14)
plt.grid(alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()

plt.show()


# Aggregate the data to calculate the average opening price per year
avg_open_per_year = df.groupby('Year')['Open'].mean()

# Generate smooth x and y values for interpolation
x = avg_open_per_year.index
y = avg_open_per_year.values
x_smooth = np.linspace(x.min(), x.max(), 500)  # Increase the number of points for smoothness
spline = make_interp_spline(x, y)
y_smooth = spline(x_smooth)

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(x_smooth, y_smooth, label='Avg Opening Price ', color='blue', linewidth=2)

# Annotate the highest and lowest points
max_year = avg_open_per_year.idxmax()
min_year = avg_open_per_year.idxmin()
max_value = avg_open_per_year.max()
min_value = avg_open_per_year.min()

plt.scatter(max_year, max_value, color='green', label='Highest Avg Opening Price', zorder=5)
plt.scatter(min_year, min_value, color='red', label='Lowest Avg Opening Price', zorder=5)

# Add annotations
plt.text(max_year, max_value, f'{max_value:.2f}', color='green', fontsize=10, ha='center', va='bottom')
plt.text(min_year, min_value, f'{min_value:.2f}', color='red', fontsize=10, ha='center', va='top')

# Customize the plot
plt.title('Average Opening Price Per Year', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Avg Opening Price', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()


df['Profit/Loss'] = df['Close'].diff()
df['Year'] = df.index.year
avg_profit_loss_per_year = df.groupby('Year')['Profit/Loss'].mean()
print(avg_profit_loss_per_year)
df['Year'] = df.index.year

# Compute yearly profit or loss
yearly_profit_loss = df.groupby('Year').agg(
    Start_Open=('Open', 'first'),  # First Open price of the year
    End_Close=('Close', 'last')   # Last Close price of the year
)
yearly_profit_loss['Profit/Loss'] = yearly_profit_loss['End_Close'] - yearly_profit_loss['Start_Open']

# Plot the yearly profit or loss
plt.figure(figsize=(12, 6))
plt.bar(
    yearly_profit_loss.index,
    yearly_profit_loss['Profit/Loss'],
    color=['green' if x > 0 else 'red' for x in yearly_profit_loss['Profit/Loss']],
    alpha=0.7
)
yearly_profit_loss['Profit/Loss (%)'] = (
    (yearly_profit_loss['End_Close'] - yearly_profit_loss['Start_Open']) / yearly_profit_loss['Start_Open']
) * 100

plt.axhline(0, color='black', linestyle='--', linewidth=0.8)  # Add a horizontal line at y=0
plt.title('Profit or Loss Per Year (BTC)')
plt.xlabel('Year')
plt.ylabel('Profit/Loss (USD)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
