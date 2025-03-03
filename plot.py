import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

def plot_demographic_comparison(df_old_counties, df_new_counties, df_full, county_col="County", demographic_cols=None):
    """
    Create a bar plot comparing demographic distributions between two sets of counties and return detailed statistics.
    Stores absolute population counts and calculates percentages only for visualization.
    
    Parameters:
    -----------
    df_old_counties : pandas.DataFrame
        DataFrame containing the original set of counties
    df_new_counties : pandas.DataFrame
        DataFrame containing the new set of counties
    df_full : pandas.DataFrame
        DataFrame containing demographic data for all counties
    county_col : str, default="County"
        Name of the column containing county identifiers
    demographic_cols : dict, default=None
        Dictionary mapping demographic categories to their column names
        e.g. {'White': 'white_pct', 'Black': 'black_pct', ...}
        
    Returns:
    --------
    tuple
        - matplotlib.figure.Figure: Figure containing the demographic comparison plot
        - dict: Dictionary containing detailed statistics including:
            - 'old_stats': Population counts for original counties
            - 'new_stats': Population counts for new counties
            - 'differences': Absolute population differences between sets
            - 'percent_changes': Relative percent changes between sets
            - 'population_stats': Total population statistics
    """
    
    # Default demographic columns if none provided
    if demographic_cols is None:
        demographic_cols = {
            'White': 'White',
            'Black': 'Black',
            'Asian': 'Asian',
            'Hispanic': 'Hispanic',
            'Native': 'Native',
            'Pacific': 'Pacific',
            'Men': 'Men',
            'Women': 'Women'
        }
    
    def get_demographic_stats(counties_df):
        # # Merge with full data to get demographics
        # merged = df_full[df_full[county_col].isin(counties_df[county_col])]

        # # print merged where county_col = Jefferson County
        # jefferson_rows = merged[merged[county_col] == "Jefferson County"]

        # # Print the filtered rows
        # print(jefferson_rows)

        # fixed
        merged = pd.merge(df_full, counties_df, on=["County", "State"], how="inner")

        # print merged where county_col = Jefferson County
        jefferson_rows = merged[merged[county_col] == "Jefferson County"]

        # # Print the filtered rows
        # print(jefferson_rows)
        
        stats = {
            'total_population': merged['TotalPop'].sum(),
            'num_counties': len(merged),
            'demographics': {}
        }
        
        for demo, col in demographic_cols.items():
            if demo in ['Men', 'Women']:
                # For gender, sum the population counts directly
                total_in_group = merged[col].sum()
            else:
                # For racial demographics, multiply percentage by population and sum
                total_in_group = (merged[col] * merged['TotalPop'] / 100).sum().round()
            
            # Store absolute population count
            stats['demographics'][demo] = int(total_in_group)
            
        return stats
    
    # Calculate demographic statistics for both sets
    old_stats = get_demographic_stats(df_old_counties)
    new_stats = get_demographic_stats(df_new_counties)
    
    # Calculate absolute differences and percent changes
    differences = {}
    percent_changes = {}
    for demo in demographic_cols.keys():
        old_count = old_stats['demographics'][demo]
        new_count = new_stats['demographics'][demo]
        
        differences[demo] = new_count - old_count
        percent_changes[demo] = ((new_count - old_count) / old_count * 100) if old_count != 0 else float('inf')
    
    # Calculate percentages for visualization
    def get_percentages(stats):
        return {demo: (count / stats['total_population'] * 100) 
                for demo, count in stats['demographics'].items()}
    
    old_demographic_pcts = get_percentages(old_stats)
    new_demographic_pcts = get_percentages(new_stats)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(demographic_cols))
    width = 0.35
    
    old_percentages = [old_demographic_pcts[demo] for demo in demographic_cols.keys()]
    new_percentages = [new_demographic_pcts[demo] for demo in demographic_cols.keys()]
    
    # Create bars
    ax.bar([i - width/2 for i in x], old_percentages, width, 
           label='Results Before DQ Issues', color='skyblue')
    ax.bar([i + width/2 for i in x], new_percentages, width, 
           label='Results After DQ Issues', color='lightcoral')
    
    # Customize plot
    ax.set_ylabel('Percentage')
    ax.set_title('Demographic Distributions in Final Results Before and After DQ Issues')
    ax.set_xticks(x)
    ax.set_xticklabels(demographic_cols.keys())
    ax.legend()
    
    # Add percentage labels on bars
    def add_labels(positions, values):
        for pos, val in zip(positions, values):
            ax.text(pos, val, f'{val:.1f}%', ha='center', va='bottom')
    
    add_labels([i - width/2 for i in x], old_percentages)
    add_labels([i + width/2 for i in x], new_percentages)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Prepare return statistics
    stats_dict = {
        'old_stats': {
            'demographics': old_stats['demographics'],  # Now contains absolute counts
            'total_population': old_stats['total_population'],
            'num_counties': old_stats['num_counties']
        },
        'new_stats': {
            'demographics': new_stats['demographics'],  # Now contains absolute counts
            'total_population': new_stats['total_population'],
            'num_counties': new_stats['num_counties']
        },
        'differences': differences,  # Absolute population differences
        'percent_changes': percent_changes,
        'population_change': {
            'absolute': new_stats['total_population'] - old_stats['total_population'],
            'percentage': ((new_stats['total_population'] - old_stats['total_population']) / old_stats['total_population']) * 100
        }
    }
    
    return fig, stats_dict

def correlation_matrix_grid(df, x_columns=None, y_columns=None):
    """
    Create a simple correlation matrix for selected columns.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    x_columns : list or None
        Columns for x-axis. If None, uses all columns
    y_columns : list or None
        Columns for y-axis. If None, uses x_columns
    """
    # Handle columns
    if x_columns is None:
        x_columns = df.columns.tolist()
    if y_columns is None:
        y_columns = x_columns

    # Calculate correlations
    correlation_matrix = pd.DataFrame(
        [[df[x].corr(df[y]) for y in y_columns] for x in x_columns],
        index=x_columns,
        columns=y_columns
    )

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.tight_layout()

    return correlation_matrix

def correlation_matrix_feature(df, feature):
    # correlation matrix
    correlation_matrix = df.corr()[feature]
    print("Correlation Matrix:")
    print(correlation_matrix)

    # sort by highest correlation
    correlation_matrix.sort_values(ascending=False).plot(kind='bar', color='skyblue', figsize=(15, 6))
    plt.title(f"Correlations with {feature}")
    plt.ylabel("Correlation Coefficient")
    plt.xlabel("Features")
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.tight_layout()

    return plt

def plot_distribution(df, column=None):
    """
    Plots distributions for each variable in the DataFrame.

    Parameters:
    df (pd.DataFrame): Input pandas DataFrame.
    column (str): Input variable name
    """
    if column:
        plt.figure(figsize=(max(12, len(df[column].dropna().unique()) * 0.3), 6))
        plt.hist(df[column].dropna(), bins=30, edgecolor='k', alpha=0.7)
        plt.title(column)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
      # total num variables
      num_vars = len(df.columns)

      # calculate row and col size
      cols = math.ceil(math.sqrt(num_vars))
      rows = math.ceil(num_vars / cols)

      # set up figure
      fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
      axes = axes.flatten()

      # for each variable, create a plot
      for i, col in enumerate(df.columns):
          # plot histogram
          axes[i].hist(df[col].dropna(), bins=30, edgecolor='k', alpha=0.7)
          # set title as title
          axes[i].set_title(col)
          # axes lables
          axes[i].set_xlabel('Value')
          axes[i].set_ylabel('Frequency')

          # rotate axis
          axes[i].tick_params(axis='x', rotation=45)

          # adjust subplot width based on number of x labels
          unique_vals = len(df[col].dropna().unique())
          axes[i].set_xticks(axes[i].get_xticks())
          axes[i].figure.set_size_inches(max(12, unique_vals * 0.5), rows * 4)

      # remove unused subplots
      for i in range(num_vars, len(axes)):
          fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

def summarize_variable_distributions(df, column=None):
    """
    Provides a numerical summary of distributions for each variable in the DataFrame or for a specific variable if provided.

    Parameters:
    df (pd.DataFrame): Input pandas DataFrame.
    column (str, optional): Specific column to summarize. If None, summarizes all variables.
    """
    if column:
        summary = df[column].describe()
        #print(summary)
        counts = df[column].value_counts(dropna=False)
        percentages = df[column].value_counts(normalize=True, dropna=False) * 100
        summary_df = pd.DataFrame({'Count': counts, 'Percentage': percentages})
        print(summary_df)
    else:
        print("Summary for all variables:")
        for col in df.columns:
            print(f"\n{col}:")
            summary = df[col].describe()
            #print(summary)
            counts = df[col].value_counts(dropna=False)
            percentages = df[col].value_counts(normalize=True, dropna=False) * 100
            summary_df = pd.DataFrame({'Count': counts, 'Percentage': percentages})
            print(summary_df)
    print("\n")

    return summary_df

def expected_distributions(x_train, y_train, col):
  # combine train x and y
  combined_train = pd.concat([x_train, y_train], axis=1)

  # divide by target value (binary for now)
  combined_train_y1 = combined_train[combined_train['target'] == 1]
  combined_train_y0 = combined_train[combined_train['target'] == 0]

  # print distributions for train and test
  print(f"Expected Distributions for {col} (Train Distributions)")
  print("-----------------")
  print("target == 0")
  y0_summary = summarize_variable_distributions(combined_train_y0, column=col)
  print("target == 1")
  y1_summary = summarize_variable_distributions(combined_train_y1, column=col)

  return y0_summary, y1_summary

def predicted_distributions(x_test, y_pred, col):
  # combine test x and y
  combined_test = pd.concat([x_test, y_pred], axis=1)

  combined_test_y1 = combined_test[combined_test['target'] == 1]
  combined_test_y0 = combined_test[combined_test['target'] == 0]

  print(f"Predicted Distributions for {col} (Pred. Distributions)")
  print("-----------------")
  print("target == 0")
  y0_summary = summarize_variable_distributions(combined_test_y0, column=col)
  print("target == 1")
  y1_summary = summarize_variable_distributions(combined_test_y1, column=col)

  return y0_summary, y1_summary

def summarize_one_hot_distributions(df, col_prefix):
    """
    Summarizes distributions for one-hot-encoded variables that share the same prefix.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing one-hot-encoded columns.
    col_prefix (str): Prefix shared by one-hot-encoded columns.
    """
    # Select columns with the given prefix
    relevant_cols = [col for col in df.columns if col.startswith(col_prefix)]

    # Compute counts and percentages by summing values across rows
    counts = df[relevant_cols].sum()
    percentages = (counts / len(df)) * 100

    # Create summary DataFrame
    summary_df = pd.DataFrame({'Count': counts, 'Percentage': percentages})
    print(summary_df)
    print("\n")

    return summary_df


def expected_distributions_one_hot(x_train, y_train, col_prefix):
    """
    Computes expected distributions for one-hot-encoded columns grouped by target.

    Parameters:
    x_train (pd.DataFrame): Training features.
    y_train (pd.DataFrame or pd.Series): Training labels.
    col_prefix (str): Prefix shared by one-hot-encoded columns.
    """
    # Combine x and y
    combined_train = pd.concat([x_train, y_train], axis=1)

    # Split based on target values
    combined_train_y1 = combined_train[combined_train['target'] == 1]
    combined_train_y0 = combined_train[combined_train['target'] == 0]

    # Print distributions for each target
    print(f"Expected Distributions for {col_prefix} (One-Hot Encoded Train Distributions)")
    print("-----------------")
    print("target == 0")
    y0_summary = summarize_one_hot_distributions(combined_train_y0, col_prefix)
    print("target == 1")
    y1_summary = summarize_one_hot_distributions(combined_train_y1, col_prefix)

    return y0_summary, y1_summary


def predicted_distributions_one_hot(x_test, y_pred, col_prefix):
    """
    Computes expected distributions for one-hot-encoded columns grouped by target.

    Parameters:
    x_train (pd.DataFrame): Training features.
    y_train (pd.DataFrame or pd.Series): Training labels.
    col_prefix (str): Prefix shared by one-hot-encoded columns.
    """
    # Combine x and y
    combined_test = pd.concat([x_test, y_pred], axis=1)

    # Split based on target values
    combined_test_y1 = combined_test[combined_test['target'] == 1]
    combined_test_y0 = combined_test[combined_test['target'] == 0]

    # Print distributions for each target
    print(f"Predicted Distributions for {col_prefix} (One-Hot Encoded Test Distributions)")
    print("-----------------")
    print("target == 0")
    y0_summary = summarize_one_hot_distributions(combined_test_y0, col_prefix)
    print("target == 1")
    y1_summary = summarize_one_hot_distributions(combined_test_y1, col_prefix)

    return y0_summary, y1_summary

def plot_summary(df1, df2, dist_type, col, dq_type, target1=0, target2=1, show_labels=True):
   """
   Plots two dataframes side by side for comparison based on percentages and counts.
   
   Parameters:
   df1, df2 (pd.DataFrame): DataFrames containing 'Count' and 'Percentage'.
   col (str): Column name or label for x-axis.
   target1, target2 (int): Labels for the two groups being compared.
   show_labels (bool): Whether to show percentage and count labels on bars
   """
   # Get all unique categories and fill missing with 0
   all_index = df1.index.union(df2.index)
   df1 = df1.reindex(all_index, fill_value=0)
   df2 = df2.reindex(all_index, fill_value=0)
   
   # Setup figure size and bar width
   plt.figure(figsize=(14, 7))  # Made figure wider
   bar_width = 0.35  # Slightly narrower bars
   
   # Positions for bars
   r1 = range(len(df1))  # X positions for first set of bars
   r2 = [x + bar_width for x in r1]  # X positions for second set of bars
   
   # Plot bars for both DataFrames
   bars1 = plt.bar(r1, df1['Percentage'], width=bar_width, label=f'Target = {target1}', color='skyblue')
   bars2 = plt.bar(r2, df2['Percentage'], width=bar_width, label=f'Target = {target2}', color='lightcoral')
   
   # Add labels and title
   plt.xlabel(col)
   plt.ylabel('Percentage (%)')
   plt.title(f'{dist_type} Distribution of {col} for {dq_type}')
   
   # Add more space between x-axis labels
   plt.xticks([r + bar_width / 2 for r in range(len(df1))], df1.index, rotation=45, ha='right')
   plt.subplots_adjust(bottom=0.2)  # Add more space at bottom
   
   if show_labels:
       # Annotate percentages and counts on bars for df1
       for i, bar in enumerate(bars1):
           height = bar.get_height()
           count = df1['Count'].iloc[i]
           percentage = df1['Percentage'].iloc[i]
           plt.text(bar.get_x() + bar.get_width() / 2, height + 1,
                   f'{percentage:.1f}%\n({count})', ha='center', fontsize=10)
       
       # Annotate percentages and counts on bars for df2
       for i, bar in enumerate(bars2):
           height = bar.get_height()
           count = df2['Count'].iloc[i]
           percentage = df2['Percentage'].iloc[i]
           plt.text(bar.get_x() + bar.get_width() / 2, height + 1,
                   f'{percentage:.1f}%\n({count})', ha='center', fontsize=10)
   
   # Display legend and plot
   plt.legend()
   plt.tight_layout()
   plt.show()

# def plot_summary(df1, df2, dist_type, col, dq_type, target1=0, target2=1):
#     """
#     Plots two dataframes side by side for comparison based on percentages and counts.

#     Parameters:
#     df1, df2 (pd.DataFrame): DataFrames containing 'Count' and 'Percentage'.
#     col (str): Column name or label for x-axis.
#     target1, target2 (int): Labels for the two groups being compared.
#     """
#     # Setup figure size and bar width
#     plt.figure(figsize=(12, 7))
#     bar_width = 0.4  # Width of each bar

#     # Positions for bars
#     r1 = range(len(df1))  # X positions for first set of bars
#     r2 = [x + bar_width for x in r1]  # X positions for second set of bars

#     # Plot bars for both DataFrames
#     bars1 = plt.bar(r1, df1['Percentage'], width=bar_width, label=f'Target = {target1}', color='skyblue')
#     bars2 = plt.bar(r2, df2['Percentage'], width=bar_width, label=f'Target = {target2}', color='lightcoral')

#     # Add labels and title
#     plt.xlabel(col)
#     plt.ylabel('Percentage (%)')
#     plt.title(f'{dist_type} Distribution of {col} for {dq_type}')
#     plt.xticks([r + bar_width / 2 for r in range(len(df1))], df1.index, rotation=45)  # Center ticks and rotate

#     # Annotate percentages and counts on bars for df1
#     for i, bar in enumerate(bars1):
#         height = bar.get_height()
#         count = df1['Count'].iloc[i]
#         percentage = df1['Percentage'].iloc[i]
#         plt.text(bar.get_x() + bar.get_width() / 2, height + 1,
#                 f'{percentage:.1f}%\n({count})', ha='center', fontsize=10)

#     # Annotate percentages and counts on bars for df2
#     for i, bar in enumerate(bars2):
#         height = bar.get_height()
#         count = df2['Count'].iloc[i]
#         percentage = df2['Percentage'].iloc[i]
#         plt.text(bar.get_x() + bar.get_width() / 2, height + 1,
#                 f'{percentage:.1f}%\n({count})', ha='center', fontsize=10)

#     # Display legend and plot
#     plt.legend()
#     plt.tight_layout()
#     plt.show()