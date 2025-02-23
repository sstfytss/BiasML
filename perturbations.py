import numpy as np
import pandas as pd

def _is_one_hot_encoded(df, base):
    """
    Check if a column is one-hot encoded.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    base (str): Column name to check.

    Returns:
    bool: True if one-hot encoded, False otherwise.
    """

    # find cols with the base
    cols = [col for col in df.columns if col.startswith(f"{base}_")]
    # conditions for one-hot encoding:
    # 1. multiple cols exist
    # 2. cols are boolean
    # 3. each row has exactly one true
    cond1, cond2, cond3 = len(cols) > 1, all(df[col].dtype == bool for col in cols), df[cols].apply(lambda row: row.sum() == 1, axis=1).all()
    #print(cond1, cond2, cond3)

    if not cond3:
      print("here")
      invalid_rows = ~df[cols].apply(lambda row: row.sum() == 1, axis=1)
      # Show the specific rows that don't meet the one-hot encoding criteria
      #print("bad rows", len(df[invalid_rows]))

    if cond1 and cond2 and cond3:
        return True
    else:
        return False

def add_random_noise(df, prob_replace=0.2, noise_scale=0.1, noise_col=None, cond_col=None, condition_function = None):
    """
    Check if a column is one-hot encoded.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    prob_replace (float): Probability to add noise to a particular row
    noise_scale (float): Scale for the STD of noise
    noise_col (str): The column to add noise too

    Returns:
    df (pd.DataFrame): Output DataFrame with noise.
    """
    df_noisy = df.copy()

    # determine the column type
    if _is_one_hot_encoded(df_noisy, noise_col):
      print("one_hot_encoded")
      encoded_cols = [col for col in df_noisy.columns if col.startswith(f"{noise_col}_")]

      for idx in range(len(df_noisy)):
        # Find the current true column for this specific row
        current_true_col = [col for col in encoded_cols if df_noisy.loc[idx, col]][0]

        # Select a different column to make true
        other_cols = [col for col in encoded_cols if col != current_true_col]

        # if the condition is met
        if condition_function(df_noisy.loc[idx, cond_col]):
          # print("condition met: ", current_true_col)
          # draw the probability
          if np.random.rand() < prob_replace:
              # choose a new column
              new_true_col = np.random.choice(other_cols)
              # print("new true col", new_true_col)

              # flip the columns for this row
              df_noisy.loc[idx, current_true_col] = False
              df_noisy.loc[idx, new_true_col] = True
    else:
      if df_noisy[noise_col].dtype == 'object' or df_noisy[noise_col].dtype == 'category':  # categorical columns
          print("categorical")
          # Create mask for values that meet the condition
          condition_mask = df_noisy[cond_col].apply(condition_function).astype(bool)
          # Generate random probabilities only for values that meet the condition
          random_probs = np.random.rand(len(df_noisy))
          # Combined mask: value meets condition AND random prob < prob_replace
          final_mask = condition_mask & (random_probs < prob_replace)
          # randomly replace values with other values from the same column
          df_noisy.loc[final_mask, noise_col] = np.random.choice(df_noisy[noise_col], size=final_mask.sum())
      elif df_noisy[noise_col].dtype == 'int64':  # numeric columns
          # add gaussian noise
          print("int")
          # Create mask for values that meet the condition
          condition_mask = df_noisy[cond_col].apply(condition_function)
          # generate random prob
          random_probs = np.random.rand(len(df_noisy))
          # create final mask for rows that meet cond and are below prob
          final_mask = condition_mask & (random_probs < prob_replace)
          noise = np.random.normal(loc=1, scale=noise_scale * df_noisy[noise_col].std(), size=final_mask.sum())

          if final_mask.sum() > 0: # if there's more than 1 row that meets the condition
            df_noisy.loc[final_mask, noise_col] = np.round(df_noisy.loc[final_mask, noise_col] * noise).astype(int) # round back to int
      elif df_noisy[noise_col].dtype == 'float':
          # add gaussian noise
          print("float")
          # create mask on condition for cond_col
          condition_mask = df_noisy[cond_col].apply(condition_function)
          # gen random probs
          random_probs = np.random.rand(len(df_noisy))
          # create final mask for rows that meet cond and are below prob
          final_mask = condition_mask & (random_probs < prob_replace)
          noise = np.random.normal(loc=1, scale=noise_scale * df_noisy[noise_col].std(), size=final_mask.sum())

          if final_mask.sum() > 0:
            df_noisy.loc[final_mask, noise_col] = df_noisy.loc[final_mask, noise_col] * noise # round back to int
      elif df_noisy[noise_col].dtype == 'bool':  # boolean columns
          print("bool")
          condition_mask = df_noisy[cond_col].apply(condition_function)
          random_probs = np.random.rand(len(df_noisy))
          final_mask = condition_mask & (random_probs < prob_replace)

          # flip boolean values with the given probability
          df_noisy.loc[final_mask, noise_col] = ~df_noisy.loc[df_noisy, noise_col]

    return df_noisy

def add_missingness(df, prob_replace=0.2, noise_col=None, cond_col=None, condition_function=None):
    """
    Add missing values (NaN) to specified columns in a DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    prob_missing (float): Probability to set a value as missing for a particular row
    missing_col (str): The column to add missing values to
    cond_col (str): Column to check condition on
    condition_function (callable): Function that returns True/False for condition checking

    Returns:
    df (pd.DataFrame): Output DataFrame with missing values.
    """
    import numpy as np
    import pandas as pd
    
    df_missing = df.copy()

    # determine the column type
    if _is_one_hot_encoded(df_missing, noise_col):
        print("one_hot_encoded")
        encoded_cols = [col for col in df_missing.columns if col.startswith(f"{noise_col}_")]

        for idx in range(len(df_missing)):
            # Check condition if provided
            if condition_function and not condition_function(df_missing.loc[idx, cond_col]):
                continue
                
            # Only proceed with probability prob_missing
            if np.random.rand() >= prob_replace:
                continue
                
            # Set all encoded columns to 0 (equivalent to NaN for one-hot)
            for col in encoded_cols:
                df_missing.loc[idx, col] = 0
                
    else:
        # Create condition mask
        condition_mask = pd.Series(True, index=df_missing.index)
        if condition_function and cond_col:
            condition_mask = df_missing[cond_col].apply(condition_function)
        
        # Generate random probabilities
        random_mask = np.random.rand(len(df_missing)) < prob_replace
        
        # Combined mask: value meets condition AND random prob < prob_missing
        final_mask = condition_mask & random_mask
        
        # Set values to NaN regardless of dtype
        df_missing.loc[final_mask, noise_col] = np.nan
        
        # For categorical columns, ensure the column type remains categorical
        if df_missing[noise_col].dtype.name == 'category':
            df_missing[noise_col] = df_missing[noise_col].astype('category')

    return df_missing
