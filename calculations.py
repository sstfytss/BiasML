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
    if (majority_col in ['Men', 'W