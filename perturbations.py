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
    print(cond1, cond2, cond3)

    if not cond3:
      print("here")
      invalid_rows = ~df[cols].apply(lambda row: row.sum() == 1, axis=1)
      # Show the specific rows that don't meet the one-hot encoding criteria
      print("bad rows", len(df[invalid_rows]))

    if cond1 and cond2 and cond3:
        return True
    else:
        return False

def add_random_noise(df, prob_replace=0.2, prob_flip=0.3, noise_scale=0.1, noise_col=None):
    # create a copy
    df_noisy = df.copy()

    # determine the column type
    if _is_one_hot_encoded(df_noisy, noise_col):
      print("one_hot_encoded")
      encoded_cols = [col for col in df_noisy.columns if col.startswith(f"{noise_col}_")]

      for idx in range(len(df_noisy)):
        if np.random.rand() < prob_replace:
            # Find the current true column for this specific row
            current_true_col = [col for col in encoded_cols if df_noisy.loc[idx, col]][0]

            # Select a different column to make true
            other_cols = [col for col in encoded_cols if col != current_true_col]
            new_true_col = np.random.choice(other_cols)

            # Flip the columns for this row
            df_noisy.loc[idx, current_true_col] = False
            df_noisy.loc[idx, new_true_col] = True
    else:
      if df_noisy[noise_col].dtype == 'object' or df_noisy[noise_col].dtype == 'category':  # categorical columns
          print("categorical")
          # randomly replace values with other values from the same column
          mask = np.random.rand(len(df_noisy)) < prob_replace
          df_noisy.loc[mask, noise_col] = np.random.choice(df_noisy[noise_col], size=mask.sum())
      elif df_noisy[noise_col].dtype == 'int64':  # numeric columns
          # add gaussian noise
          print("int")
          mask = np.random.rand(len(df_noisy)) < prob_replace
          #noise = np.random.normal(0, noise_level * df_noisy[col].std(), size=mask.sum())
          noise = np.random.normal(loc=1, scale=noise_scale * df_noisy[noise_col].std(), size=mask.sum())
          df_noisy.loc[mask, noise_col] = np.round(df_noisy.loc[mask, noise_col] * noise).astype(int) # round back to int
      elif df_noisy[noise_col].dtype == 'float':
          # add gaussian noise
          print("float")
          mask = np.random.rand(len(df_noisy)) < prob_replace
          #noise = np.random.normal(0, noise_level * df_noisy[col].std(), size=mask.sum())
          noise = np.random.normal(loc=1, scale=noise_scale * df_noisy[noise_col].std(), size=mask.sum())
          df_noisy.loc[mask, noise_col] = df_noisy.loc[mask, noise_col] * noise # round back to int
      elif df_noisy[noise_col].dtype == 'bool':  # boolean columns
          print("bool")
          # flip boolean values with the given probability
          mask = np.random.rand(len(df_noisy)) < prob_flip
          df_noisy.loc[mask, noise_col] = ~df_noisy.loc[mask, noise_col]

    return df_noisy
