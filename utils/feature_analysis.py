import matplotlib.pyplot as plt
import seaborn as sns
def plot_occupancy(df, title):
    plt.figure(figsize=(10, 6))
    plt.plot(df['timestamp'], df['occupancy_percentage'], label='occupancy_percentage')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Occupancy Percentage')
    plt.legend()
    plt.show()

def plot_occupancy_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['occupancy_percentage'], kde=True)
    plt.title('Distribution of Gym Occupancy')
    plt.xlabel('Occupancy Percentage')
    plt.ylabel('Frequency')
    plt.show()

def plot_occupancy_by_day_of_week(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='day_of_week', y='occupancy_percentage', data=df)
    plt.title('Gym Occupancy by Day of the Week')
    plt.xlabel('Day of the Week')
    plt.ylabel('Occupancy Percentage')
    plt.show()
#plot heatmap of occupancy by time and day of the week
def plot_heatmap_occupancy_by_time_and_day(df):
    pivot = df.pivot_table(index='hour', columns='day_of_week', values='occupancy_percentage', aggfunc='mean')
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, cmap='YlGnBu', annot=True, fmt=".2f")
    plt.title('Gym Occupancy by Time and Day of the Week')
    plt.xlabel('Day of the Week')
    plt.ylabel('Time of Day')
    plt.show()

# plot correlation matrix
def plot_correlation_matrix(df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()

# plot occupancy by weather condition
def plot_occupancy_by_weather_condition(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='weather_condition', y='occupancy_percentage', data=df)
    plt.title('Gym Occupancy by Weather Condition')
    plt.xlabel('Weather Condition')
    plt.ylabel('Occupancy Percentage')
    plt.show()

#plot temperature by occupancy
def plot_temperature_by_occupancy(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='temperature', y='occupancy_percentage', data=df)
    plt.title('Temperature vs Gym Occupancy')
    plt.xlabel('Temperature')
    plt.ylabel('Occupancy Percentage')
    plt.show()


# load processed data
df = pd.read_csv('data/processed_gym_occupancy_data.csv')

plot_occupancy_distribution(df)
# plot_occupancy(df, 'Gym Occupancy Over Time')
plot_occupancy_by_day_of_week(df)
plot_heatmap_occupancy_by_time_and_day(df)
plot_occupancy_by_weather_condition(df)
plot_temperature_by_occupancy(df)
plot_correlation_matrix(df)
