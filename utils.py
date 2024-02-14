import yaml
import pandas as pd

def read_yaml_to_dict(filepath: str) -> dict:
    """
    Reads a YAML file and converts it to a Python dictionary.

    Parameters:
    - filepath: str, path to the YAML file.

    Returns:
    - dict: The contents of the YAML file as a Python dictionary.
    """
    with open(filepath, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")
            return None

# def convert_incoming_loc_to_proper_df(df: pd.DataFrame, misgrabs_or_collisions: str, lim: int, fryer_slot_id: int, feature_dummy_dict: dict) -> pd.DataFrame:

def create_slot_df(df: pd.DataFrame, misgrabs_or_collisions: str, lim: int, fryer_slot_id: int, feature_dummy_dict: dict) -> pd.DataFrame:
    """remove dups, nans, filter for either misgrabs or collisions only, remove unrelated errors (that are not coll/mis)
    ensure the df is sorted by behavior_start_time in asc order, remove all but N most recent rows
    add in error_code_0.0 and error_code_1.0, make sure that this is only for a certain fryer_slot_id
    """

    #remove rows with nans and duplicate rows
    new_df = df.dropna().drop_duplicates()
    df = df[(df['fryer_slot_id'] == fryer_slot_id)]
    df['behavior_start_time'] = pd.to_datetime(df['behavior_start_time'])
    df = df.sort_values(by='behavior_start_time', ascending=True).tail(lim)


    df['error_code_1.0'] = 0
    df['error_code_0.0'] = 1

    #if we select collisions only, then set error_code_1.0 for misgrabs to 0 and error_code_0.0 to 1
    if misgrabs_or_collisions == 'misgrabs':
        df.loc[df['error_code'] == 100, 'error_code_1.0'] = 0
        df.loc[df['error_code'] == 100, 'error_code_0.0'] = 1
        df.loc[df['error_code'] == 103, 'error_code_1.0'] = 1
        df.loc[df['error_code'] == 103, 'error_code_0.0'] = 0
        df['error_code_1.0'] = df['error_code_1.0'].astype(int)  # or .astype(int) if appropriate
        df['error_code_0.0'] = df['error_code_0.0'].astype(int)  # or .astype(int) if appropriate

    #if we select collisions only, then set error_code_1.0 for misgrabs to 0 and error_code_0.0 to 1
    elif misgrabs_or_collisions == 'collisions':
        df.loc[df['error_code'] == 103, 'error_code_1.0'] = 0
        df.loc[df['error_code'] == 103, 'error_code_0.0'] = 1
        df.loc[df['error_code'] == 100, 'error_code_1.0'] = 1
        df.loc[df['error_code'] == 100, 'error_code_0.0'] = 0
        df['error_code_1.0'] = df['error_code_1.0'].astype(int)  # or .astype(int) if appropriate
        df['error_code_0.0'] = df['error_code_0.0'].astype(int)  # or .astype(int) if appropriate
    #Remove unrelated errors
    df = df[(df['error_code'] == 100) | (df['error_code'] == 103) | (df['error_code'] == 0)]

    #Add dummies:
    for source_col, categorical_cols in feature_dummy_dict.items():
        add_categoricals(df, categorical_cols, source_col)

    return df


def rename_columns(df: pd.DataFrame, original_columns: list[str], new_columns: list[str]) -> pd.DataFrame:
    """
    Rename specified columns in a DataFrame.

    Parameters:
    - df: The DataFrame whose columns are to be renamed.
    - original_columns: A list of original column names to be renamed.
    - new_columns: A list of new column names.

    Returns:
    - A DataFrame with the specified columns renamed.
    """
    # Ensure the lengths of the original and new columns lists match
    if len(original_columns) != len(new_columns):
        raise ValueError("The list of original columns and new columns must have the same length.")

    # Create a mapping from original column names to new column names
    rename_mapping = dict(zip(original_columns, new_columns))

    # Rename the columns and return the modified DataFrame
    return df.rename(columns=rename_mapping)


def combine_single_row_dfs(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    print(len(df1), len(df2))
    print(df1.head())
    print(df2.head())
    
    # Ensure both DataFrames have only one row
    if len(df1) != 1 or len(df2) != 1:
        raise ValueError("Both DataFrames must have exactly one row.")
    
    # Convert the single-row DataFrames to Series to flatten them
    series1 = df1.iloc[0]
    series2 = df2.iloc[0]
    
    # Combine the Series, giving priority to df1 in case of overlapping columns
    combined_series = pd.concat([series1, series2[~series2.index.isin(series1.index)]])
    
    # Convert the combined Series back to a single-row DataFrame
    combined_df = pd.DataFrame([combined_series])
    
    return combined_df


def prepare_input_data(slot_df: pd.DataFrame, input_localization_data: pd.DataFrame, window_list: list[int], col_to_avg_list: list[str], basket_id: int, fryer_slot_id: int, basket_state: str, feature_dummy_dict: dict, cols_to_rename: list[str], cols_to_rename_to: list[str], col_to_drop_early: list[str], site_id: str, site_id_possibilities: list[str]) -> pd.DataFrame:
    
    input_localization_data = rename_columns(input_localization_data, cols_to_rename, cols_to_rename_to) 

    for source_col, categorical_cols in feature_dummy_dict.items():
        add_categoricals(input_localization_data, categorical_cols, source_col)

    rolling_avg_df_list = []

    for window in window_list:
        rolling_avg_df_list.append(add_rolling_means_for_specific_criteria(slot_df, col_to_avg_list, window, fryer_slot_id, basket_id, basket_state))

    combined_df = pd.concat(rolling_avg_df_list, axis=1)
    combined_df2 = combine_single_row_dfs(input_localization_data, combined_df)
    drop_cols(combined_df2, col_to_drop_early)
    add_site_id_categorical(combined_df2, site_id, site_id_possibilities)
    return combined_df2


def drop_cols(df, cols_to_drop):
    return df.drop(columns=cols_to_drop, inplace=True)

def add_categoricals(df, categorical_cols, source_col):
    # Iterate through each categorical column
    for col in categorical_cols:
        # Check if the value in source_col is a substring of the column name
        # Set the column value to 1 if true, else 0
        df[col] = df[source_col].apply(lambda x: 1 if str(x) in str(col) else 0)


def add_rolling_means_for_specific_criteria(df: pd.DataFrame, columns_to_normalize: list[str], window_size: int, fryer_slot_id: int, basket_id: int, basket_state: str) -> pd.DataFrame:
    # Filter the DataFrame and check if the result is empty
    filtered_df = df[(df['fryer_slot_id'] == fryer_slot_id) & 
                     (df['basket_id'] == basket_id) & 
                     (df['basket_state'] == basket_state)]

    if filtered_df.empty:
        print("No data matches the given criteria.")
        return pd.DataFrame()  # Return an empty DataFrame or handle as needed

    # Initialize an empty DataFrame to store the new columns
    new_df = pd.DataFrame()

    # Calculate the mean for the last `window_size` rows (or fewer) for each specified column
    for column in columns_to_normalize:
        if pd.api.types.is_numeric_dtype(filtered_df[column]):
            # Adjust window size if necessary
            actual_window_size = min(len(filtered_df), window_size)
            mean_value = filtered_df[column].tail(actual_window_size).mean()
            new_df[f'{column}_mean_since_beginning_window_{window_size}'] = [mean_value]
        else:
            print(f"Warning: Column '{column}' is not numeric and was skipped.")

    return new_df



def add_site_id_categorical(df, site_id, site_id_possibilities):
    for site in site_id_possibilities:
        if site_id in site:
            df[site] = int(1)
        else:
            df[site] = int(0)

def get_n_random_input_rows_from_csv(n, csv_path):
    df = pd.read_csv(csv_path)
    row_lst = []
    for _ in range(n):
        row_lst.append(df.sample(n=1))
    return row_lst



# df_to_test = pd.read_csv(CSV_PATH) #IGNORE


#PAIR DOWN SLOT DATAFRAME, ADD COLS FOR CATEGORIES

# new_df = create_slot_df(df_to_test, 'misgrabs', 50000, 4, FEATURE_DUMMY_DICT)

#RENAME DATAFRAME COLS IN SLOT DF
# new_df = rename_columns(new_df, COLS_TO_RENAME, COLS_TO_RENAME_TO) 


#PICK RANDOM SAMPLE INPUT ROW
# input_df = pd.read_csv(CSV_PATH).sample(n=1)


# combined_df = prepare_input_data(new_df, input_df, AVG_WINDOWS, ec_col, 56, 4, 'frying', FEATURE_DUMMY_DICT, COLS_TO_RENAME, COLS_TO_RENAME_TO, COL_TO_DROP_EARLY, "wc26")
