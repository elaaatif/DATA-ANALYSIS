import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data from CSV
df = pd.read_csv('HPI_master.csv')

# Identify numerical and non-numerical columns
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
non_numerical_columns = df.select_dtypes(exclude=['float64', 'int64']).columns

# Univariate analysis for numerical columns
for column in numerical_columns:
    descriptive_stats = df[column].describe()
    mean_value = descriptive_stats['mean']
    median_value = df[column].median()
    mode_value = df[column].mode()[0]
    range_value = df[column].max() - df[column].min()
    variance_value = df[column].var()
    std_deviation_value = df[column].std()

    print(f'Descriptive Statistics for {column}:\n{descriptive_stats}\n')
    print(f'Mean(Moy): {mean_value}')
    print(f'Median: {median_value}')
    print(f'Mode: {mode_value}')
    print(f'Range: {range_value}')
    print(f'Variance: {variance_value}')
    print(f'Standard Deviation: {std_deviation_value}\n')

    # Histogram
    plt.hist(df[column], bins='auto', alpha=0.7)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

    # Box plot
    sns.boxplot(x=df[column])
    plt.title(f'Box Plot of {column}')
    plt.show()

    # Density plot
    sns.kdeplot(df[column], fill=True)
    plt.title(f'Density Plot of {column}')
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.show()

    # Frequency table and cumulative frequency table
    frequency_table = pd.DataFrame(df[column].value_counts())
    cumulative_frequency_table = frequency_table.cumsum()

    print(f'Frequency Table for {column}:\n{frequency_table}')
    print(f'\nCumulative Frequency Table for {column}:\n{cumulative_frequency_table}\n')

# Univariate analysis for non-numerical columns
for column in non_numerical_columns:
    frequency_table = pd.DataFrame(df[column].value_counts())

    print(f'Frequency Table for {column}:\n{frequency_table}\n')

    # Plotting bar plot for non-numerical data
    plt.figure(figsize=(10, 5))
    sns.countplot(x=column, data=df)
    plt.title(f'Bar Plot of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.show()
