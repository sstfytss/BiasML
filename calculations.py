import pandas as pd

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

def binary_changes(df_old, df_new, id_col="County"):
    """
    Compare two dataframes and identify false positives, false negatives, true positives, and true negatives
    based on the presence or absence of entries in each dataframe.
    
    Parameters:
    -----------
    df_old : pandas.DataFrame
        The reference/original dataframe
    df_new : pandas.DataFrame
        The comparison/new dataframe
    id_col : str, default="County"
        The column name to use as the identifier for matching entries
        
    Returns:
    --------
    tuple of pandas.DataFrame:
        false_positives: Entries that exist in df_new but not in df_old
        false_negatives: Entries that exist in df_old but not in df_new
        true_positives: Entries that exist in both dataframes
        true_negatives: Entries that don't exist in either dataframe (empty dataframe)
    """

    # create sets of unique identifiers (counties) from each df
    old_ids = set(df_old[id_col])
    new_ids = set(df_new[id_col])
    
    # fp: in new but not in old
    false_positive_ids = new_ids - old_ids
    false_positives = df_new[df_new[id_col].isin(false_positive_ids)]
    
    # fn: in old but not in new
    false_negative_ids = old_ids - new_ids
    false_negatives = df_old[df_old[id_col].isin(false_negative_ids)]
    
    # tp: in both
    true_positive_ids = old_ids.intersection(new_ids)
    true_positives = df_new[df_new[id_col].isin(true_positive_ids)]
    
    # tn: in neither (always empty)
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
    if (majority_col in ['Men', 'Women'] and minority_col not in ['Men', 'Women']) or (minority_col in ['Men', 'Women'] and majority_col not in ['Men', 'Women']):
      raise ValueError("Gender and race mix not allowed")
    elif (majority_col in ['White', 'Black', 'Asian', 'Pacific', 'Native', 'Hispanic'] and minority_col not in ['White', 'Black', 'Asian', 'Pacific', 'Native', 'Hispanic']) or (minority_col in ['White', 'Black', 'Asian', 'Pacific', 'Native', 'Hispanic'] and majority_col not in ['White', 'Black', 'Asian', 'Pacific', 'Native', 'Hispanic']):
      raise ValueError("Gender and race mix not allowed")
    
    # the data is stored differently for gender and race
    if majority_col in ['Men', 'Women'] or minority_col in ['Men', 'Women']:
      agg_data = (full_data.groupby(id_col)
                  .agg({
                      # get populations of each demographic
                      population_col: 'sum',
                      majority_col: lambda x: (x).sum(),
                      minority_col: lambda x: (x).sum()
                  })
                  .reset_index())
    else:
      agg_data = (full_data.groupby(id_col)
                  .agg({
                      # get populations of each demographic
                      population_col: 'sum',
                      majority_col: lambda x: (x * 0.01 * full_data.loc[x.index, population_col]).sum().round(),
                      minority_col: lambda x: (x * 0.01 * full_data.loc[x.index, population_col]).sum().round()
                  })
                  .reset_index())
    
    # join demographic data with the tp, fn dataframes
    tp_data = true_positives[[id_col]].merge(agg_data, on=id_col, how='left')
    fn_data = false_negatives[[id_col]].merge(agg_data, on=id_col, how='left')

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
    
    return bias_metric, metrics, tp_data, fn_data
