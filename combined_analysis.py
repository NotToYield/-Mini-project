import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from textblob import TextBlob

def load_data(file_path):
    """Load the dataset from the CSV file."""
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except FileNotFoundError:
        print("File not found. Please check the file path.")
        exit()

def explore_data(data):
    """Display basic statistics and info about the dataset."""
    print("\nData Summary:")
    print(data.info())
    
    print("\nDescriptive Statistics:")
    print(data.describe())
    
    print("\nFirst few rows of data:")
    print(data.head())

def plot_variable_distribution(data):
    """Plot the distribution of a selected variable."""
    print("Available variables:", data.columns)
    var = input("Enter the variable to plot: ")
    
    if var in data.columns:
        sns.histplot(data[var], kde=True)
        plt.title(f"Distribution of {var}")
        plt.xlabel(var)
        plt.ylabel("Frequency")
        plt.show()
    else:
        print("Invalid column name.")

def perform_anova(data):
    """Conduct ANOVA on selected variables."""
    print("Available variables:", data.columns)
    
    # Choose variables
    continuous_var = input("Enter a continuous (interval/ratio) variable: ")
    categorical_var = input("Enter a categorical (ordinal/nominal) variable: ")
    
    if continuous_var in data.columns and categorical_var in data.columns:
        # Visual check for normality
        sns.boxplot(x=categorical_var, y=continuous_var, data=data)
        plt.title(f"Boxplot of {continuous_var} by {categorical_var}")
        plt.show()
        
        print("Performing ANOVA...")
        anova_result = stats.f_oneway(*[data[continuous_var][data[categorical_var] == cat] for cat in data[categorical_var].unique()])
        print(f"ANOVA result: F-statistic = {anova_result.statistic}, p-value = {anova_result.pvalue}")
    else:
        print("Invalid column names. Please try again.")

def perform_regression(data):
    """Perform linear regression and visualize the relationship."""
    x_var = input("Enter the independent variable: ")
    y_var = input("Enter the dependent variable: ")
    
    if x_var in data.columns and y_var in data.columns:
        X = sm.add_constant(data[x_var])
        y = data[y_var]
        
        model = sm.OLS(y, X).fit()
        print(model.summary())
        
        # Plot the regression line and scatter plot
        sns.regplot(x=x_var, y=y_var, data=data, line_kws={"color": "red"})
        plt.title(f"Regression: {y_var} vs {x_var}")
        plt.show()
    else:
        print("Invalid column names.")

def perform_ttest(data):
    """Perform a t-test between two groups and visualize the comparison."""
    group_var = input("Enter the grouping variable: ")
    test_var = input("Enter the test variable: ")
    
    if group_var in data.columns and test_var in data.columns:
        group1 = data[test_var][data[group_var] == data[group_var].unique()[0]]
        group2 = data[test_var][data[group_var] == data[group_var].unique()[1]]
        
        t_stat, p_val = stats.ttest_ind(group1, group2)
        print(f"t-Test result: t-statistic = {t_stat}, p-value = {p_val}")
        
        # Visualize the group comparison using boxplot or violin plot
        sns.boxplot(x=group_var, y=test_var, data=data)
        plt.title(f"{test_var} by {group_var} - t-Test")
        plt.show()
    else:
        print("Invalid column names.")

def perform_chi_square(data):
    """Perform a chi-square test for independence and visualize the result."""
    cat_var1 = input("Enter the first categorical variable: ")
    cat_var2 = input("Enter the second categorical variable: ")
    
    if cat_var1 in data.columns and cat_var2 in data.columns:
        contingency_table = pd.crosstab(data[cat_var1], data[cat_var2])
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        print(f"Chi-Square result: chi2 = {chi2}, p-value = {p}")
        
        # Visualize the contingency table using a heatmap
        sns.heatmap(contingency_table, annot=True, cmap="coolwarm", cbar=True)
        plt.title(f"Contingency Table: {cat_var1} vs {cat_var2}")
        plt.show()
    else:
        print("Invalid column names.")

def analyze_sentiment(data):
    """Perform sentiment analysis if there is text data."""
    print("Searching for text data...")
    
    text_column = None
    for col in data.columns:
        if data[col].dtype == 'object':  # Looking for a text column
            text_column = col
            break
    
    if text_column:
        print(f"Analyzing sentiment for text data in column: {text_column}")
        data['Sentiment'] = data[text_column].apply(lambda text: TextBlob(text).sentiment.polarity)
        print(data[[text_column, 'Sentiment']].head())
    else:
        print("No suitable text data found for sentiment analysis.")
