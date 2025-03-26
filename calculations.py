import statistics
import pandas as pd

def get_percentiles(df, column_name):
    """
    Calculate specific percentiles (10th, 25th, 50th, 75th, 90th, 95th) for a DataFrame column.

    Parameters:
    df (pd.DataFrame): Input DataFrame
    column_name (str): Name of the column to calculate percentiles for

    Returns:
    dict: Dictionary containing the percentiles and their values
    """
    percentiles = [10, 15, 20, 25, 50, 75, 90, 95]
    results = df[column_name].describe(percentiles=[0.10, 0.15, 0.20, 0.25, 0.50, 0.75, 0.90, 0.95])

    return {
        'p10': results['10%'],
        'p15': results['15%'],
        'p20': results['20%'],
        'p25': results['25%'],
        'p50': results['50%'],
        'p75': results['75%'],
        'p90': results['90%'],
        'p95': results['95%']
    }


def rank_changes(df_old, df_new, id_col="County"):
    """
    Compares the rankings (row positions) of counties between the original and data quality issue results.

    Parameters:
      df_old (pd.DataFrame): The first dataframe (e.g. older ranking).
      df_new (pd.DataFrame): The second dataframe (e.g. new ranking).
      id_col (str): The name of the column that uniquely identifies a row (default "county").

    Returns:
      increased (pd.DataFrame): Counties that improved in ranking (i.e. moved up).
      decreased (pd.DataFrame): Counties that fell in ranking (i.e. moved down).

    Note:
      A county is considered to have improved if its new row index is lower than its old row index.
    """
    # reset the index so index is ranking position
    df_old = df_old.reset_index(drop=True)
    df_new = df_new.reset_index(drop=True)

    # create df of rankings from old and new df
    # row index is the rank (0 is the best, 1 is next, etc.)
    rank_old = pd.DataFrame({
        id_col: df_old[id_col],
        'old_rank': df_old.index
    })
    rank_new = pd.DataFrame({
        id_col: df_new[id_col],
        'new_rank': df_new.index
    })

    # merge ranking dfs on unique id (need to use outer since there may be mismatch for LIMIT dfs)
    ranks = pd.merge(rank_old, rank_new, on=id_col, how="outer",)
    print(ranks.head())

    # compute difference in rank (pos diff means rank improved)
    # ex: if old_rank is 5 and new_rank is 3, then diff = 5 - 3 = 2, meaning jump up
    ranks['rank_diff'] = ranks['old_rank'] - ranks['new_rank']

    print(ranks.head())

    # counties that improved (moved up): new_rank is lower than old_rank.
    increased = ranks[ranks['rank_diff'] > 0].copy()
    # counties that fell (moved down): new_rank is higher than old_rank.
    decreased = ranks[ranks['rank_diff'] < 0].copy()

    # Optionally, sort the results by the magnitude of the change.
    increased = increased.sort_values(by='rank_diff', ascending=False)
    decreased = decreased.sort_values(by='rank_diff')

    return increased, decreased

def binary_changes(df_old, df_new, id_col="County", state_col="State"):
    """
    Compare two dataframes and identify false positives, false negatives, true positives, and true negatives
    based on the presence or absence of entries in each dataframe. This version uses both a county and state
    for matching entries.
    
    Parameters:
    -----------
    df_old : pandas.DataFrame
        The reference/original dataframe.
    df_new : pandas.DataFrame
        The comparison/new dataframe.
    id_col : str, default="County"
        The column name to use as the county identifier.
    state_col : str, default="State"
        The column name to use as the state identifier.
        
    Returns:
    --------
    tuple of pandas.DataFrame:
        false_positives: Entries that exist in df_new but not in df_old (based on county and state)
        false_negatives: Entries that exist in df_old but not in df_new (based on county and state)
        true_positives: Entries that exist in both dataframes (based on county and state)
        true_negatives: Entries that don't exist in either dataframe (empty dataframe)
    """
    
    # Create sets of tuples (county, state) for each DataFrame.
    old_keys = set(zip(df_old[id_col], df_old[state_col]))
    new_keys = set(zip(df_new[id_col], df_new[state_col]))
    
    # Determine keys for false positives, false negatives, and true positives.
    false_positive_keys = new_keys - old_keys
    false_negative_keys = old_keys - new_keys
    true_positive_keys = old_keys.intersection(new_keys)

    print("false negative keys", false_negative_keys)
    print("false positive keys", false_positive_keys)
    
    # Filter the DataFrames using a row-wise lambda that constructs a (county, state) tuple.
    false_positives = df_new[df_new.apply(lambda row: (row[id_col], row[state_col]) in false_positive_keys, axis=1)]
    false_negatives = df_old[df_old.apply(lambda row: (row[id_col], row[state_col]) in false_negative_keys, axis=1)]
    true_positives = df_new[df_new.apply(lambda row: (row[id_col], row[state_col]) in true_positive_keys, axis=1)]
    
    # There are no true negatives by this definition.
    true_negatives = pd.DataFrame(columns=df_old.columns)
    
    return false_positives, false_negatives, true_positives, true_negatives


def binary_disparate_impact(true_positives, true_negatives, false_positives, false_negatives,
                                   full_data, id_col, majority_col, minority_col, 
                                   population_col="TotalPop"):
    """
    Calculate bias metric comparing true positive rates between demographic groups,
    using total population counts for each demographic group.
    
    Parameters:
    -----------
    true_positives : pandas.DataFrame
        DataFrame containing true positive cases (with id_col)
    true_negatives : pandas.DataFrame
        DataFrame containing true negative cases (with id_col)
    false_positives : pandas.DataFrame
        DataFrame containing false positive cases (with id_col)
    false_negatives : pandas.DataFrame
        DataFrame containing false negative cases (with id_col)
    full_data : pandas.DataFrame
        Complete dataset containing demographic and population information
    id_col : str
        Column name identifying unique entries (e.g., 'County')
    majority_col : str
        Column name containing percentage of majority group (e.g., 'white', 'male')
    minority_col : str
        Column name containing percentage of minority group (e.g., 'black', 'female')
    population_col : str, default="TotalPop"
        Column name containing total population counts
        
    Returns:
    --------
    float:
        Population-weighted bias metric
    dict:
        Additional metrics including TPR for each group and population counts
    """

    # throw error for gender and race mix
    state_col = "State"
    if (majority_col in ['Men', 'Women'] and minority_col not in ['Men', 'Women']) or (minority_col in ['Men', 'Women'] and majority_col not in ['Men', 'Women']):
      raise ValueError("Gender and race mix not allowed")
    elif (majority_col in ['White', 'Black', 'Asian', 'Pacific', 'Native', 'Hispanic'] and minority_col not in ['White', 'Black', 'Asian', 'Pacific', 'Native', 'Hispanic']) or (minority_col in ['White', 'Black', 'Asian', 'Pacific', 'Native', 'Hispanic'] and majority_col not in ['White', 'Black', 'Asian', 'Pacific', 'Native', 'Hispanic']):
      raise ValueError("Gender and race mix not allowed")
    
    # the data is stored differently for gender and race
    if majority_col in ['Men', 'Women'] or minority_col in ['Men', 'Women']:
      agg_data = (full_data.groupby([id_col, state_col])
                  .agg({
                      # get populations of each demographic
                      population_col: 'sum',
                      majority_col: lambda x: (x).sum(),
                      minority_col: lambda x: (x).sum()
                  })
                  .reset_index())
    else:
      agg_data = (full_data.groupby([id_col, state_col])
                  .agg({
                      # get populations of each demographic
                      population_col: 'sum',
                      majority_col: lambda x: (x * 0.01 * full_data.loc[x.index, population_col]).sum().round(),
                      minority_col: lambda x: (x * 0.01 * full_data.loc[x.index, population_col]).sum().round()
                  })
                  .reset_index())
    
    # join demographic data with the tp, fn dataframes
    tp_data = true_positives[[id_col, state_col]].merge(agg_data, on=[id_col, state_col], how='left')
    fn_data = false_negatives[[id_col, state_col]].merge(agg_data, on=[id_col, state_col], how='left')

    # calculate tpr for majority group
    majority_tp_pop = tp_data[majority_col].sum()
    majority_fn_pop = fn_data[majority_col].sum()
    majority_total_pop = majority_tp_pop + majority_fn_pop
    majority_tpr = majority_tp_pop / majority_total_pop if majority_total_pop > 0 else 0
    
    # calculate tpr for minority group
    minority_tp_pop = tp_data[minority_col].sum()
    minority_fn_pop = fn_data[minority_col].sum()
    minority_total_pop = minority_tp_pop + minority_fn_pop
    minority_tpr = minority_tp_pop / minority_total_pop if minority_total_pop > 0 else 0
    
    # calculate bias metric
    bias_metric = minority_tpr / majority_tpr if majority_tpr > 0 else float('inf')
    
    # compile
    metrics = {
        'minority_tpr': minority_tpr,
        'majority_tpr': majority_tpr,
        'minority_total_population': minority_total_pop,
        'majority_total_population': majority_total_pop,
        'minority_tp_population': minority_tp_pop,
        'majority_tp_population': majority_tp_pop,
        'minority_fn_population': minority_fn_pop,
        'majority_fn_population': majority_fn_pop,
        'total_counties': len(agg_data),
        'tp_counties': len(tp_data),
        'fn_counties': len(fn_data)
    }

    print(bias_metric)
    print(metrics)
    
    return bias_metric, metrics

def process_iteration_results(bias_metrics, stats_list):
    """
    Process bias metrics and demographic percentages across iterations.
    First aggregates total counts, then calculates percentages and their standard deviations.

    Parameters:
    bias_metrics (list): List of bias metric values from each iteration.
    stats_list (list): List of stats dictionaries from each iteration.

    Returns:
    tuple: ((average_bias, bias_std), (demographic_percentages, demographic_std))
        - average_bias: float representing the average bias metric.
        - bias_std: float representing the standard deviation of the bias metric.
        - demographic_percentages: dict containing aggregated demographic percentages for old and new stats.
        - demographic_std: dict containing the standard deviation for each demographic percentage across iterations.
    """
    # calc average bias and its std
    avg_bias = sum(bias_metrics) / len(bias_metrics)
    bias_std = statistics.stdev(bias_metrics) if len(bias_metrics) > 1 else 0

    # list of demographics
    demographics = ['White', 'Black', 'Asian', 'Hispanic', 'Native', 'Pacific', 'Men', 'Women']

    # aggregate counts for percentages (weighted over all iterations)
    old_counts = {demo: 0 for demo in demographics}
    new_counts = {demo: 0 for demo in demographics}
    old_total = 0
    new_total = 0

    # list for percentages (to compute standard deviation)
    new_percentages_list = {demo: [] for demo in demographics}

    # for each iterations stats
    for stats in stats_list:
        old_stats = stats['old_stats']
        new_stats = stats['new_stats']

        # update aggregated counts and total populations.
        for demo in demographics:
            old_counts[demo] += old_stats['demographics'][demo]
            new_counts[demo] += new_stats['demographics'][demo]
        old_total += old_stats['total_population']
        new_total += new_stats['total_population']

        # here compute the actual percentage for the iteration (so we can calculate the STD later)
        for demo in demographics:
            if new_stats['total_population']: # we use new_stats['total_pop'] bc this is the stat dictionary for the particular iteration
                new_percent = (new_stats['demographics'][demo] / new_stats['total_population']) * 100
            else:
                new_percent = 0

            # store new percentages in a list
            new_percentages_list[demo].append(new_percent)

    # compute the aggregated statistics for each demographic
    demographic_percentages = {
        'old_stats': {
            demo: (count / old_total) * 100 if old_total else 0
            for demo, count in old_counts.items()
        },
        'new_stats': {
            demo: (count / new_total) * 100 if new_total else 0
            for demo, count in new_counts.items()
        }
    }

    # calc standard deviation over all iterations (just for DF with DQ issues)
    demographic_std = {
        'old_stats': {},
        'new_stats': {}
    }
    for demo in demographics:
        if len(new_percentages_list[demo]) > 1:
            demographic_std['new_stats'][demo] = statistics.stdev(new_percentages_list[demo])
        else:
            demographic_std['new_stats'][demo] = 0

    return avg_bias, demographic_percentages, bias_std, demographic_std