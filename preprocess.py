import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df, y):
    """
    Preprocesses the dataset by encoding categorical variables and handling missing values.

    Parameters:
    df (pd.DataFrame): Input pandas DataFrame.

    Returns:
    pd.DataFrame: Processed DataFrame.
    """

    # make a copy of x and y
    df_copy = df.copy()
    y_copy = y.copy()

    # combine x and y
    df_copy['target'] = y_copy

    # drop dups
    df_copy = df_copy.drop_duplicates()
    # drop nans
    df_copy = df_copy.dropna()

    # native-country: remove all rows where the value is "?"
    df_copy = df_copy.loc[df_copy['native-country'] != "?"]
    df_copy = df_copy.loc[df_copy['workclass'] != "?"]
    df_copy = df_copy.loc[df_copy['occupation'] != "?"]


    # reassign target
    df_copy['target'] = df_copy['target'].str.strip()
    df_copy['target'] = df_copy['target'].replace({'<=50K.': '<=50K', '>50K.': '>50K'})
    print("targets:", df_copy['target'].unique())

    # encode
    le = LabelEncoder()
    df_copy['target'] = le.fit_transform(df_copy['target'])
    print("mappings:", le.classes_)

    # seperate x and y
    y_copy = df_copy['target']
    df_copy = df_copy.drop('target', axis=1)

    # INDIVIDUAL COL. PRE-PROCESSING
    # age
    df_copy['age'] = df_copy['age'].astype(int)

    # education is ordinal
    education_order = {
        'Preschool': 0,
        '1st-4th': 1,
        '5th-6th': 2,
        '7th-8th': 3,
        '9th': 4,
        '10th': 5,
        '11th': 6,
        '12th': 7,
        'HS-grad': 8,
        'Some-college': 9,
        'Assoc-voc': 10,
        'Assoc-acdm': 11,
        'Bachelors': 12,
        'Masters': 13,
        'Doctorate': 14,
        'Prof-school': 15
    }
    df_copy['education'] = df_copy['education'].map(education_order)
    df_copy['education'] = df_copy['education'].astype('category')


    print(len(df_copy["native-country"].unique()))

    # handle categorical variables
    label_encoders = {}
    for col in df_copy.select_dtypes(include=['object']).columns:
        # if binary, encode with 1 or 0
        if df_copy[col].nunique() == 2:
            print("categorical (binary):", col)
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col])
            label_encoders[col] = le
            print("mappings:", le.classes_)

            # let the column by category type
            df_copy[col] = df_copy[col].astype('category')
        else: # otherwise, one hot encode
            print("nominal:", col)
            df_copy = pd.get_dummies(df_copy, columns=[col], drop_first=False)

    # reset index
    df_copy = df_copy.reset_index(drop=True)
    y_copy = y_copy.reset_index(drop=True)

    return df_copy, y_copy