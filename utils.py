import yaml
import pandas as pd

import sys

# Set maximum line width to maximum integer value
sys.display_width = sys.maxsize

# Set maximum line count to maximum integer value
sys.display_height = sys.maxsize

# Set maximum item count to maximum integer value
sys.display_limit = sys.maxsize


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
    # del(df)
    
    df = new_df
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




FEATURE_DUMMY_DICT = {
    "behavior_name": [
        'behavior_name_fryer_to_dump_to_fill_to_fryer',
        'behavior_name_fryer_to_dump_to_hanger',
        'behavior_name_fryer_to_hanger',
        'behavior_name_fryer_to_outrack',
        'behavior_name_fryer_to_tune_fryer_to_fryer',
        'behavior_name_fryer_to_tune_grab_to_fryer',
        'behavior_name_hanger_to_fill_to_fryer',
        'behavior_name_hanger_to_fryer',
        'behavior_name_hanger_to_outrack',
        'behavior_name_hanger_to_tune_grab_to_hanger',
        'behavior_name_hanger_to_tune_hanger_to_hanger',
    ],
    'fryer_slot_id': [
        'fryer_slot_id_1',
        'fryer_slot_id_2',
        'fryer_slot_id_3',
        'fryer_slot_id_4',
        'fryer_slot_id_5',
        'fryer_slot_id_6',
        'fryer_slot_id_7',
        'fryer_slot_id_8',
    ],
    'basket_id': [
        'basket_id_50',
        'basket_id_51',
        'basket_id_52',
        'basket_id_53',
        'basket_id_54',
        'basket_id_55',
        'basket_id_56',
        'basket_id_57',
        'basket_id_58',
        'basket_id_59',
        'basket_id_110',
        'basket_id_111',
        'basket_id_112',
        'basket_id_113',
        'basket_id_114',
        'basket_id_115',
        'basket_id_116',
        'basket_id_117',
        'basket_id_118',
        'basket_id_119',
    ],
    'basket_state': [
        'basket_state_frying',
        'basket_state_hanging',
    ]
}

ec_col = [
    'error_code_0.0',
    'error_code_1.0',
    'basket_source_x',
    'basket_source_y',
    'basket_source_z',
    'basket_source_roll',
    'basket_source_pitch',
    'basket_source_yaw',
    'fryer_slot_x',
    'fryer_slot_y',
    'fryer_slot_z',
    'fryer_slot_roll',
    'fryer_slot_pitch',
    'fryer_slot_yaw',
    'hanger_tool_slot_x',
    'hanger_tool_slot_y',
    'hanger_tool_slot_z',
    'hanger_tool_slot_roll',
    'hanger_tool_slot_pitch',
    'hanger_tool_slot_yaw',
    'basket_source_no_offset_x',
    'basket_source_no_offset_y',
    'basket_source_no_offset_z',
    'basket_source_no_offset_roll',
    'basket_source_no_offset_pitch',
    'basket_source_no_offset_yaw',
    'fryer_slot_no_offset_x',
    'fryer_slot_no_offset_y',
    'fryer_slot_no_offset_z',
    'fryer_slot_no_offset_roll',
    'fryer_slot_no_offset_pitch',
    'fryer_slot_no_offset_yaw',
    'hanger_tool_slot_no_offset_x',
    'hanger_tool_slot_no_offset_y',
    'hanger_tool_slot_no_offset_z',
    'hanger_tool_slot_no_offset_roll',
    'hanger_tool_slot_no_offset_pitch',
    'hanger_tool_slot_no_offset_yaw',
    
    'fryer_target_abs_offset_x',
    'fryer_target_abs_offset_y',
    'fryer_target_abs_offset_z',
    'fryer_target_rel_offset_x',
    'fryer_target_rel_offset_y',
    'fryer_target_rel_offset_z',
    'hanger_target_abs_offset_x',
    'hanger_target_abs_offset_y',
    'hanger_target_abs_offset_z',
    'hanger_target_rel_offset_x',
    'hanger_target_rel_offset_y',
    'hanger_target_rel_offset_z',
    'basket_source_abs_offset_x',
    'basket_source_abs_offset_y',
    'basket_source_abs_offset_z',
    'basket_source_rel_offset_x',
    'basket_source_rel_offset_y',
    'basket_source_rel_offset_z',
]



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

COLS_TO_RENAME = [
    "failure_description",
    "fryer_slot_x_offset_abs",
    "fryer_slot_x_offset_rel",
    "fryer_slot_y_offset_abs",
    "fryer_slot_y_offset_rel",
    "fryer_slot_z_offset_abs",
    "fryer_slot_z_offset_rel",
    "hanger_tool_slot_x_offset_abs",
    "hanger_tool_slot_x_offset_rel",
    "hanger_tool_slot_y_offset_abs",
    "hanger_tool_slot_y_offset_rel",
    "hanger_tool_slot_z_offset_abs",
    "hanger_tool_slot_z_offset_rel",
    "basket_source_x_offset_abs",
    "basket_source_x_offset_rel",
    "basket_source_y_offset_abs",
    "basket_source_y_offset_rel",
    "basket_source_z_offset_abs",
    "basket_source_z_offset_rel",
    "fryer_slot_x_no_offset",
    "fryer_slot_y_no_offset",
    "fryer_slot_z_no_offset",
    "fryer_slot_roll_no_offset",
    "fryer_slot_pitch_no_offset",
    "fryer_slot_yaw_no_offset",
    "hanger_tool_slot_x_no_offset",
    "hanger_tool_slot_y_no_offset",
    "hanger_tool_slot_z_no_offset",
    "hanger_tool_slot_roll_no_offset",
    "hanger_tool_slot_pitch_no_offset",
    "hanger_tool_slot_yaw_no_offset",
    "basket_source_x_no_offset",
    "basket_source_y_no_offset",
    "basket_source_z_no_offset",
    "basket_source_roll_no_offset",
    "basket_source_pitch_no_offset",
    "basket_source_yaw_no_offset",
 ]

COLS_TO_RENAME_TO = [
    "Failure Description",
    "fryer_target_abs_offset_x",
    "fryer_target_rel_offset_x",
    "fryer_target_abs_offset_y",
    "fryer_target_rel_offset_y",
    "fryer_target_abs_offset_z",
    "fryer_target_rel_offset_z",
    "hanger_target_abs_offset_x",
    "hanger_target_rel_offset_x",
    "hanger_target_abs_offset_y",
    "hanger_target_rel_offset_y",
    "hanger_target_abs_offset_z",
    "hanger_target_rel_offset_z",
    "basket_source_abs_offset_x",
    "basket_source_rel_offset_x",
    "basket_source_abs_offset_y",
    "basket_source_rel_offset_y",
    "basket_source_abs_offset_z",
    "basket_source_rel_offset_z",
    "fryer_slot_no_offset_x",
    "fryer_slot_no_offset_y",
    "fryer_slot_no_offset_z",
    "fryer_slot_no_offset_roll",
    "fryer_slot_no_offset_pitch",
    "fryer_slot_no_offset_yaw",
    "hanger_tool_slot_no_offset_x",
    "hanger_tool_slot_no_offset_y",
    "hanger_tool_slot_no_offset_z",
    "hanger_tool_slot_no_offset_roll",
    "hanger_tool_slot_no_offset_pitch",
    "hanger_tool_slot_no_offset_yaw",
    "basket_source_no_offset_x",
    "basket_source_no_offset_y",
    "basket_source_no_offset_z",
    "basket_source_no_offset_roll",
    "basket_source_no_offset_pitch",
    "basket_source_no_offset_yaw",
]




COLS_TO_AVG = [
    'error_code_0.0',
    'error_code_1.0',

    "fryer_target_abs_offset_x",
    "fryer_target_rel_offset_x",
    "fryer_target_abs_offset_y",
    "fryer_target_rel_offset_y",
    "fryer_target_abs_offset_z",
    "fryer_target_rel_offset_z",
    "hanger_target_abs_offset_x",
    "hanger_target_rel_offset_x",
    "hanger_target_abs_offset_y",
    "hanger_target_rel_offset_y",
    "hanger_target_abs_offset_z",
    "hanger_target_rel_offset_z",
    "basket_source_abs_offset_x",
    "basket_source_rel_offset_x",
    "basket_source_abs_offset_y",
    "basket_source_rel_offset_y",
    "basket_source_abs_offset_z",
    "basket_source_rel_offset_z",
    "fryer_slot_no_offset_x",
    "fryer_slot_no_offset_y",
    "fryer_slot_no_offset_z",
    "fryer_slot_no_offset_roll",
    "fryer_slot_no_offset_pitch",
    "fryer_slot_no_offset_yaw",
    "hanger_tool_slot_no_offset_x",
    "hanger_tool_slot_no_offset_y",
    "hanger_tool_slot_no_offset_z",
    "hanger_tool_slot_no_offset_roll",
    "hanger_tool_slot_no_offset_pitch",
    "hanger_tool_slot_no_offset_yaw",
    "basket_source_no_offset_x",
    "basket_source_no_offset_y",
    "basket_source_no_offset_z",
    "basket_source_no_offset_roll",
    "basket_source_no_offset_pitch",
    "basket_source_no_offset_yaw",
]

AVG_WINDOWS = [
    1,
    2,
    11,
    51,
    101,
    250,
    1001,
]

COL_TO_DROP_EARLY = [
    "Unnamed: 0",
    "exec_success",
    "behavior_name",
    "fryer_slot_id",
    "basket_id",
    "behavior_start_time",
    "error_code",
    "Failure Description",
    "basket_state",
]
# def combine_single_row_dfs(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
#     # Ensure both DataFrames have only one row
#     if len(df1) != 1 or len(df2) != 1:
#         raise ValueError("Both DataFrames must have exactly one row.")
    
#     # Combine the DataFrames, giving priority to df1 in case of overlapping columns
#     combined_df = pd.concat([df1, df2], axis=1)
    
#     # Drop duplicate columns, keeping the first occurrence (from df1)
#     combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
    
#     return combined_df

def combine_single_row_dfs(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
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


def prepare_input_data(slot_df: pd.DataFrame, input_localization_data: pd.DataFrame, feature_cols: list, cols_to_drop: list, window_list: list[int], col_to_avg_list: list[str], basket_id: int, fryer_slot_id: int, basket_state: str, feature_dummy_dict: dict, cols_to_rename: list[str], cols_to_rename_to: list[str], col_to_drop_early: list[str]) -> pd.DataFrame:
    
    input_localization_data = rename_columns(input_localization_data, cols_to_rename, cols_to_rename_to) 



    #Add dummies to input_localization_data df:
    for source_col, categorical_cols in feature_dummy_dict.items():
        add_categoricals(input_localization_data, categorical_cols, source_col)

    
    # drop_cols(input_localization_data, COL_TO_DROP_EARLY)

    # print(f"len of slot df: {len(slot_df)}")
    # print(f"slot df head: \n\n\n {slot_df[(slot_df['basket_id'] == 50) ]}")# & (slot_df['fryer_slot_id'] == 1)]}")
    # print(slot_df)
    rolling_avg_df_list = []
    for window in window_list:
        rolling_avg_df_list.append(add_rolling_means_for_specific_criteria(slot_df, col_to_avg_list, window, fryer_slot_id, basket_id, basket_state))
    # for df in rolling_avg_df_list:
        # for col in df.columns:
        #     print(col)
        # print("\n\n")
        # print(df.head())
        # print("\n\n\n")

    combined_df = pd.concat(rolling_avg_df_list, axis=1)

    # print("\n\nafter adding the averages combined data:\n\n")
    # print(combined_df.head())

    # combined_df = pd.concat([input_localization_data, combined_df], axis=1)

    combined_df2 = combine_single_row_dfs(input_localization_data, combined_df)
    # print(f"\n\ninput loc: {input_localization_data}")

    # print(f"\n\ncombined_df before: {combined_df2}")

 

    drop_cols(combined_df2, col_to_drop_early)

    add_site_id_categorical(combined_df2, 'jb93')

    # for col in combined_df2.columns:
    #         print(col)



    return combined_df2





def drop_cols(df, cols_to_drop):
    return df.drop(columns=cols_to_drop, inplace=True)

def add_categoricals(df, categorical_cols, source_col):
    # Iterate through each categorical column
    for col in categorical_cols:
        # Check if the value in source_col is a substring of the column name
        # Set the column value to 1 if true, else 0
        df[col] = df[source_col].apply(lambda x: 1 if str(x) in str(col) else 0)





# def add_rolling_means_for_specific_criteria(df: pd.DataFrame, columns_to_normalize: list[str], window_size: int, fryer_slot_id: int, basket_id: int, basket_state: str) -> pd.DataFrame:
#     # Filter the DataFrame for the specified criteria
#     filtered_df = df[(df['fryer_slot_id'] == fryer_slot_id) & 
#                      (df['basket_id'] == basket_id) & 
#                      (df['basket_state'] == basket_state)]#.sort_values('behavior_start_time') #SORT UNNECESSARY

#     # filtered_df = df
#     # Initialize an empty DataFrame to store the new columns
#     new_df = pd.DataFrame()

#     if len(filtered_df) >= window_size:
#         # Select the last N rows including the current row
#         last_n_rows = filtered_df.tail(window_size)
        
#         # Calculate the mean for the specified columns and add them to the new DataFrame
#         for column in columns_to_normalize:
#             mean_value = last_n_rows[column].mean()
#             new_df[f'{column}_mean_since_beginning_window_{window_size}'] = [mean_value]

#     # Ensure the new DataFrame has the correct number of columns, even if no calculations were done
#     if new_df.empty:
#         for column in columns_to_normalize:
#             new_df[f'{column}_mean_since_beginning_window_{window_size}'] = [pd.NA]

#     return new_df

# import pandas as pd

# def add_rolling_means_for_specific_criteria(df: pd.DataFrame, columns_to_normalize: list[str], window_size: int, fryer_slot_id: int, basket_id: int, basket_state: str) -> pd.DataFrame:
#     # Filter the DataFrame for the specified criteria
#     filtered_df = df[(df['fryer_slot_id'] == fryer_slot_id) & 
#                      (df['basket_id'] == basket_id) & 
#                      (df['basket_state'] == basket_state)]

#     # Calculate rolling mean for each specified column and add it as a new column
#     for column in columns_to_normalize:
#         # The `min_periods=1` argument ensures that the mean is calculated for as few as 1 row
#         filtered_df[f'{column}_mean_since_beginning_window_{window_size}'] = filtered_df[column].rolling(window=window_size, min_periods=1).mean()

#     return filtered_df


# def add_rolling_means_for_specific_criteria(df: pd.DataFrame, columns_to_normalize: list[str], window_size: int, fryer_slot_id: int, basket_id: int, basket_state: str) -> pd.DataFrame:
#     # Filter the DataFrame and create a copy to avoid SettingWithCopyWarning
#     filtered_df = df[(df['fryer_slot_id'] == fryer_slot_id) & 
#                      (df['basket_id'] == basket_id) & 
#                      (df['basket_state'] == basket_state)].copy()

#     # Initialize an empty DataFrame to store the new columns
#     new_df = pd.DataFrame(index=[0])  # DataFrame with a single row
#     # Calculate the mean for the last `window_size` rows (or fewer) for each specified column
#     for column in columns_to_normalize:
#         # Ensure the column is numeric to avoid NotImplementedError and DataError
#         if pd.api.types.is_numeric_dtype(filtered_df[column]):
#             mean_value = filtered_df[column].tail(window_size).mean()
#             new_df[f'{column}_mean_since_beginning_window_{window_size}'] = [mean_value]
#         else:
#             # Optionally handle non-numeric columns differently, e.g., skip or raise a warning
#             print(f"Warning: Column '{column}' is not numeric and was skipped.")

#     return new_df



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


FEATURE_COLS_TO_DROP_FROM_INCOMING_LOCALIZATION_DATA = [
    'behavior_name', #CATEGORICAL, NOT INCL DIRECTLY IN FEATURES
    'fryer_slot_id', #CATEGORICAL, NOT INCL DIRECTLY IN FEATURES
    'basket_id', #CATEGORICAL, NOT INCL DIRECTLY IN FEATURES
    'basket_state', #CATEGORICAL, NOT INCL DIRECTLY IN FEATURES
]

FEATURE_COLS_TO_DROP_FROM_SLOT_DF = [
    'behavior_name', #CATEGORICAL, NOT INCL DIRECTLY IN FEATURES
    'fryer_slot_id', #CATEGORICAL, NOT INCL DIRECTLY IN FEATURES
    'basket_id', #CATEGORICAL, NOT INCL DIRECTLY IN FEATURES
    'basket_state', #CATEGORICAL, NOT INCL DIRECTLY IN FEATURES
    'behavior_start_time',
    'error_code', #NOT INCL IN FEATURES
    'exec_success', #NOT INCL IN FEATURES
    'failure_description', #NOT INCLUDE IN FEATURES
]


def add_site_id_categorical(df, site_id):
    
    site_id_possibilities = [
        'site_id_jb93',
        'site_id_wc26',
        'site_id_wc74',
        'site_id_wccg',
        'site_id_wcchic101',
        'site_id_wcchic103',
        'site_id_wcchic30',
        'site_id_wcchic42',
        'site_id_wcdetr30',
        'site_id_wcdetr36',
        'site_id_wcstls44',
        'site_id_wcstls50',
        'site_id_wcstls63',
        'site_id_wcstls65',
    ]

    for site in site_id_possibilities:
        if site_id in site:
            df[site] = int(1)
        else:
            df[site] = int(0)














CSV_PATH = "/home/sam/Downloads/behaviors_df.csv"
MODEL_PATH = "/home/sam/Downloads/misgrab_prediction/models/test_rename"
EXPECTED_COLUMNS = []
ENGINEER_FEATURES_CONFIG_FILE_PATH = "/home/sam/Downloads/misgrab_prediction/package_model2/engineer_features_config.yaml"

df_to_test = pd.read_csv(CSV_PATH)
# print("len of df to test:  ", len(df_to_test))
# print(f"len of df to test: {len(df_to_test)}")
# print(f"df_to_test head: \n\n\n {df_to_test[(df_to_test['basket_id'] == 50) ]}")# & (slot_df['fryer_slot_id'] == 1)]}")
# print(df_to_test)



# print("\n\ndf_to_test_cols:\n\n")
# for col in df_to_test.columns:
#     print(col)

pd.set_option('display.max_rows', 1000)  # Adjust the number of rows to display
pd.set_option('display.max_columns', 1000)  # Adjust the number of columns to display
pd.set_option('display.max_colwidth', 10000)  # Adjust the column width to display full content

new_df = create_slot_df(df_to_test, 'misgrabs', 50000, 4, FEATURE_DUMMY_DICT)

# print(new_df[['failure_description','error_code', 'error_code_1.0', 'error_code_0.0']][new_df['failure_description'] != 'no_failure'].head(300))
# new_df = new_df.sample(n=15)
# print(new_df.sample(n=10))


# def prepare_input_data(slot_df: pd.DataFrame, input_localization_data: pd.DataFrame, feature_cols: list, cols_to_drop: list, window_list: list[int], col_to_avg_list: list[str], basket_id: int, fryer_slot_id: int, basket_state: str):

# print("\n\nbefore:\n\n")
# # for col in new_df.columns:
# #     print(col)
# # print(new_df.dtypes)
# print(new_df.head(1))

new_df = rename_columns(new_df, COLS_TO_RENAME, COLS_TO_RENAME_TO) 
# def rename_columns(df: pd.DataFrame, original_columns: list[str], new_columns: list[str]) -> pd.DataFrame:
# print("len of new_df after rename\n\n:  ", len(new_df))
# print(f"new df: \n\n {print(new_df.iloc[:, :10])}")


# print("\n\nafter:\n\n")
# # print(new_df.dtypes)
# print(new_df.head(1))

# for col in new_df.columns:
#     print(col)

# print("\n\nnew df after renaming:\n\n")
# for col in new_df.columns:
#     print(col)

# def prepare_input_data(slot_df: pd.DataFrame, input_localization_data: pd.DataFrame, feature_cols: list, cols_to_drop: list, window_list: list[int], col_to_avg_list: list[str], basket_id: int, fryer_slot_id: int, basket_state: str, feature_dummy_dict: dict, cols_to_rename: list[str], cols_to_rename_to: list[str]):


input_df = pd.read_csv(CSV_PATH).sample(n=1)
# print(input_df.head())
# print(new_df.head())
# combined_df = prepare_input_data(new_df, input_df, [], [], AVG_WINDOWS, COLS_TO_AVG, int(input_df['basket_id']), int(input_df['fryer_slot_id']), str(input_df['basket_state']), FEATURE_DUMMY_DICT, COLS_TO_RENAME, COLS_TO_RENAME_TO, COL_TO_DROP_EARLY)

# combined_df = prepare_input_data(new_df, input_df, [], [], AVG_WINDOWS, ec_col, int(input_df['basket_id']), int(input_df['fryer_slot_id']), str(input_df['basket_state']), FEATURE_DUMMY_DICT, COLS_TO_RENAME, COLS_TO_RENAME_TO, COL_TO_DROP_EARLY)


combined_df = prepare_input_data(new_df, input_df, [], [], AVG_WINDOWS, ec_col, 56, 4, 'frying', FEATURE_DUMMY_DICT, COLS_TO_RENAME, COLS_TO_RENAME_TO, COL_TO_DROP_EARLY)


# for col in combined_df.columns:
#     print(col)
# combined_df.head(1)
first_row_as_list = combined_df.head(1).values.tolist()

# print(first_row_as_list)


def print_first_row_values(df: pd.DataFrame):
    # Iterate through the columns of the DataFrame
    for column in df.columns:
        # Get the value from the first row for the current column
        value = df.iloc[0][column]

        # value = df.at[0, column]
        # Print the column name and value
        print(f"{column}: {value}")

# Example usage:
# Assuming 'df' is your DataFrame
# print_first_row_values(combined_df)


#cols in slot_df:
slot_df_cols = [
    'behavior_name', #CATEGORICAL, NOT INCL DIRECTLY IN FEATURES
    'fryer_slot_id', #CATEGORICAL, NOT INCL DIRECTLY IN FEATURES
    'basket_id', #CATEGORICAL, NOT INCL DIRECTLY IN FEATURES
    'basket_state', #CATEGORICAL, NOT INCL DIRECTLY IN FEATURES
    'basket_source_x',
    'basket_source_y',
    'basket_source_z',
    'basket_source_roll',
    'basket_source_pitch',
    'basket_source_yaw',
    'fryer_slot_x',
    'fryer_slot_y',
    'fryer_slot_z',
    'fryer_slot_roll',
    'fryer_slot_pitch',
    'fryer_slot_yaw',
    'hanger_tool_slot_x',
    'hanger_tool_slot_y',
    'hanger_tool_slot_z',
    'hanger_tool_slot_roll',
    'hanger_tool_slot_pitch',
    'hanger_tool_slot_yaw',
    'behavior_start_time',
    'error_code', #NOT INCL IN FEATURES
    'exec_success', #NOT INCL IN FEATURES
    'failure_description', #NOT INCLUDE IN FEATURES
    'fryer_slot_x_offset_abs',
    'fryer_slot_x_offset_rel',
    'fryer_slot_y_offset_abs',
    'fryer_slot_y_offset_rel',
    'fryer_slot_z_offset_abs',
    'fryer_slot_z_offset_rel',
    'hanger_tool_slot_x_offset_abs',
    'hanger_tool_slot_x_offset_rel',
    'hanger_tool_slot_y_offset_abs',
    'hanger_tool_slot_y_offset_rel',
    'hanger_tool_slot_z_offset_abs',
    'hanger_tool_slot_z_offset_rel',
    'basket_source_x_offset_abs',
    'basket_source_x_offset_rel',
    'basket_source_y_offset_abs',
    'basket_source_y_offset_rel',
    'basket_source_z_offset_abs',
    'basket_source_z_offset_rel',
    'fryer_slot_x_no_offset',
    'fryer_slot_y_no_offset',
    'fryer_slot_z_no_offset',
    'fryer_slot_roll_no_offset',
    'fryer_slot_pitch_no_offset',
    'fryer_slot_yaw_no_offset',
    'hanger_tool_slot_x_no_offset',
    'hanger_tool_slot_y_no_offset',
    'hanger_tool_slot_z_no_offset',
    'hanger_tool_slot_roll_no_offset',
    'hanger_tool_slot_pitch_no_offset',
    'hanger_tool_slot_yaw_no_offset',
    'basket_source_x_no_offset',
    'basket_source_y_no_offset',
    'basket_source_z_no_offset',
    'basket_source_roll_no_offset',
    'basket_source_pitch_no_offset',
    'basket_source_yaw_no_offset',
]

incoming_localization_data_cols = [
    'behavior_name', #CATEGORICAL, NOT INCL DIRECTLY IN FEATURES
    'fryer_slot_id', #CATEGORICAL, NOT INCL DIRECTLY IN FEATURES
    'basket_id', #CATEGORICAL, NOT INCL DIRECTLY IN FEATURES
    'basket_state', #CATEGORICAL, NOT INCL DIRECTLY IN FEATURES
    'basket_source_x',
    'basket_source_y',
    'basket_source_z',
    'basket_source_roll',
    'basket_source_pitch',
    'basket_source_yaw',
    'fryer_slot_x',
    'fryer_slot_y',
    'fryer_slot_z',
    'fryer_slot_roll',
    'fryer_slot_pitch',
    'fryer_slot_yaw',
    'hanger_tool_slot_x',
    'hanger_tool_slot_y',
    'hanger_tool_slot_z',
    'hanger_tool_slot_roll',
    'hanger_tool_slot_pitch',
    'hanger_tool_slot_yaw',
    'behavior_start_time',
    'fryer_slot_x_offset_abs',
    'fryer_slot_x_offset_rel',
    'fryer_slot_y_offset_abs',
    'fryer_slot_y_offset_rel',
    'fryer_slot_z_offset_abs',
    'fryer_slot_z_offset_rel',
    'hanger_tool_slot_x_offset_abs',
    'hanger_tool_slot_x_offset_rel',
    'hanger_tool_slot_y_offset_abs',
    'hanger_tool_slot_y_offset_rel',
    'hanger_tool_slot_z_offset_abs',
    'hanger_tool_slot_z_offset_rel',
    'basket_source_x_offset_abs',
    'basket_source_x_offset_rel',
    'basket_source_y_offset_abs',
    'basket_source_y_offset_rel',
    'basket_source_z_offset_abs',
    'basket_source_z_offset_rel',
    'fryer_slot_x_no_offset',
    'fryer_slot_y_no_offset',
    'fryer_slot_z_no_offset',
    'fryer_slot_roll_no_offset',
    'fryer_slot_pitch_no_offset',
    'fryer_slot_yaw_no_offset',
    'hanger_tool_slot_x_no_offset',
    'hanger_tool_slot_y_no_offset',
    'hanger_tool_slot_z_no_offset',
    'hanger_tool_slot_roll_no_offset',
    'hanger_tool_slot_pitch_no_offset',
    'hanger_tool_slot_yaw_no_offset',
    'basket_source_x_no_offset',
    'basket_source_y_no_offset',
    'basket_source_z_no_offset',
    'basket_source_roll_no_offset',
    'basket_source_pitch_no_offset',
    'basket_source_yaw_no_offset',

]


feature_cols = [
    # 'behavior_name', #NOT DIRECTLY INCL IN FEATURES
    # 'site_id', #NOT DIRECTLY INCL IN FEATURES
    # 'behavior_start_time', #NOT DIRECTLY INCL IN FEATURES
    # 'basket_id', #NOT DIRECTLY INCL IN FEATURES
    # 'error_code', #NOT INCL IN FEATURES AT ALL
    # 'basket_state', #NOT DIRECTLY INCL IN FEATURES
    # 'fryer_slot_id', #NOT DIRECTLY INCL IN FEATURES
    # 'Failure Description', #NOT INCL IN FEATURES AT ALL
    'fryer_target_abs_offset_x',
    'fryer_target_abs_offset_y',
    'fryer_target_abs_offset_z',
    'fryer_target_rel_offset_x',
    'fryer_target_rel_offset_y',
    'fryer_target_rel_offset_z',
    'hanger_target_abs_offset_x',
    'hanger_target_abs_offset_y',
    'hanger_target_abs_offset_z',
    'hanger_target_rel_offset_x',
    'hanger_target_rel_offset_y',
    'hanger_target_rel_offset_z',
    'basket_source_abs_offset_x',
    'basket_source_abs_offset_y',
    'basket_source_abs_offset_z',
    'basket_source_rel_offset_x',
    'basket_source_rel_offset_y',
    'basket_source_rel_offset_z',

    'basket_source_x',
    'basket_source_y',
    'basket_source_z',
    'basket_source_roll',
    'basket_source_pitch',
    'basket_source_yaw',
    'fryer_slot_x',
    'fryer_slot_y',
    'fryer_slot_z',
    'fryer_slot_roll',
    'fryer_slot_pitch',
    'fryer_slot_yaw',
    'hanger_tool_slot_x',
    'hanger_tool_slot_y',
    'hanger_tool_slot_z',
    'hanger_tool_slot_roll',
    'hanger_tool_slot_pitch',
    'hanger_tool_slot_yaw',
    'basket_source_no_offset_x',
    'basket_source_no_offset_y',
    'basket_source_no_offset_z',
    'basket_source_no_offset_roll',
    'basket_source_no_offset_pitch',
    'basket_source_no_offset_yaw',
    'fryer_slot_no_offset_x',
    'fryer_slot_no_offset_y',
    'fryer_slot_no_offset_z',
    'fryer_slot_no_offset_roll',
    'fryer_slot_no_offset_pitch',
    'fryer_slot_no_offset_yaw',
    'hanger_tool_slot_no_offset_x',
    'hanger_tool_slot_no_offset_y',
    'hanger_tool_slot_no_offset_z',
    'hanger_tool_slot_no_offset_roll',
    'hanger_tool_slot_no_offset_pitch',
    'hanger_tool_slot_no_offset_yaw',
    'site_id_jb93',
    'site_id_wc26',
    'site_id_wc74',
    'site_id_wccg',
    'site_id_wcchic101',
    'site_id_wcchic103',
    'site_id_wcchic30',
    'site_id_wcchic42',
    'site_id_wcdetr30',
    'site_id_wcdetr36',
    'site_id_wcstls44',
    'site_id_wcstls50',
    'site_id_wcstls63',
    'site_id_wcstls65',
    'fryer_slot_id_1',
    'fryer_slot_id_2',
    'fryer_slot_id_3',
    'fryer_slot_id_4',
    'fryer_slot_id_5',
    'fryer_slot_id_6',
    'fryer_slot_id_7',
    'fryer_slot_id_8',
    'basket_id_50',
    'basket_id_51',
    'basket_id_52',
    'basket_id_53',
    'basket_id_54',
    'basket_id_55',
    'basket_id_56',
    'basket_id_57',
    'basket_id_58',
    'basket_id_59',
    'basket_id_110',
    'basket_id_111',
    'basket_id_112',
    'basket_id_113',
    'basket_id_114',
    'basket_id_115',
    'basket_id_116',
    'basket_id_117',
    'basket_id_118',
    'basket_id_119',
    # 'error_code_0.0', #NOT INCL IN FEATURES AT ALL
    # 'error_code_1.0', #NOT INCL IN FEATURES AT ALL
    'basket_state_frying',
    'basket_state_hanging',
    'behavior_name_fryer_to_dump_to_fill_to_fryer', #this one
    'behavior_name_fryer_to_dump_to_hanger', #this one
    'behavior_name_fryer_to_hanger',
    'behavior_name_fryer_to_outrack',
    'behavior_name_fryer_to_tune_fryer_to_fryer',
    'behavior_name_fryer_to_tune_grab_to_fryer',
    'behavior_name_hanger_to_fill_to_fryer',#this one
    'behavior_name_hanger_to_fryer',
    'behavior_name_hanger_to_outrack',
    'behavior_name_hanger_to_tune_grab_to_hanger',
    'behavior_name_hanger_to_tune_hanger_to_hanger',
    'error_code_0.0_mean_since_beginning_window_1',
    'error_code_1.0_mean_since_beginning_window_1',
    'basket_source_x_mean_since_beginning_window_1',
    'basket_source_y_mean_since_beginning_window_1',
    'basket_source_z_mean_since_beginning_window_1',
    'basket_source_roll_mean_since_beginning_window_1',
    'basket_source_pitch_mean_since_beginning_window_1',
    'basket_source_yaw_mean_since_beginning_window_1',
    'fryer_slot_x_mean_since_beginning_window_1',
    'fryer_slot_y_mean_since_beginning_window_1',
    'fryer_slot_z_mean_since_beginning_window_1',
    'fryer_slot_roll_mean_since_beginning_window_1',
    'fryer_slot_pitch_mean_since_beginning_window_1',
    'fryer_slot_yaw_mean_since_beginning_window_1',
    'hanger_tool_slot_x_mean_since_beginning_window_1',
    'hanger_tool_slot_y_mean_since_beginning_window_1',
    'hanger_tool_slot_z_mean_since_beginning_window_1',
    'hanger_tool_slot_roll_mean_since_beginning_window_1',
    'hanger_tool_slot_pitch_mean_since_beginning_window_1',
    'hanger_tool_slot_yaw_mean_since_beginning_window_1',
    'basket_source_no_offset_x_mean_since_beginning_window_1',
    'basket_source_no_offset_y_mean_since_beginning_window_1',
    'basket_source_no_offset_z_mean_since_beginning_window_1',
    'basket_source_no_offset_roll_mean_since_beginning_window_1',
    'basket_source_no_offset_pitch_mean_since_beginning_window_1',
    'basket_source_no_offset_yaw_mean_since_beginning_window_1',
    'fryer_slot_no_offset_x_mean_since_beginning_window_1',
    'fryer_slot_no_offset_y_mean_since_beginning_window_1',
    'fryer_slot_no_offset_z_mean_since_beginning_window_1',
    'fryer_slot_no_offset_roll_mean_since_beginning_window_1',
    'fryer_slot_no_offset_pitch_mean_since_beginning_window_1',
    'fryer_slot_no_offset_yaw_mean_since_beginning_window_1',
    'hanger_tool_slot_no_offset_x_mean_since_beginning_window_1',
    'hanger_tool_slot_no_offset_y_mean_since_beginning_window_1',
    'hanger_tool_slot_no_offset_z_mean_since_beginning_window_1',
    'hanger_tool_slot_no_offset_roll_mean_since_beginning_window_1',
    'hanger_tool_slot_no_offset_pitch_mean_since_beginning_window_1',
    'hanger_tool_slot_no_offset_yaw_mean_since_beginning_window_1',
    'fryer_target_abs_offset_x_mean_since_beginning_window_1',
    'fryer_target_abs_offset_y_mean_since_beginning_window_1',
    'fryer_target_abs_offset_z_mean_since_beginning_window_1',
    'fryer_target_rel_offset_x_mean_since_beginning_window_1',
    'fryer_target_rel_offset_y_mean_since_beginning_window_1',
    'fryer_target_rel_offset_z_mean_since_beginning_window_1',
    'hanger_target_abs_offset_x_mean_since_beginning_window_1',
    'hanger_target_abs_offset_y_mean_since_beginning_window_1',
    'hanger_target_abs_offset_z_mean_since_beginning_window_1',
    'hanger_target_rel_offset_x_mean_since_beginning_window_1',
    'hanger_target_rel_offset_y_mean_since_beginning_window_1',
    'hanger_target_rel_offset_z_mean_since_beginning_window_1',
    'basket_source_abs_offset_x_mean_since_beginning_window_1',
    'basket_source_abs_offset_y_mean_since_beginning_window_1',
    'basket_source_abs_offset_z_mean_since_beginning_window_1',
    'basket_source_rel_offset_x_mean_since_beginning_window_1',
    'basket_source_rel_offset_y_mean_since_beginning_window_1',
    'basket_source_rel_offset_z_mean_since_beginning_window_1',
    'error_code_0.0_mean_since_beginning_window_2',
    'error_code_1.0_mean_since_beginning_window_2',
    'basket_source_x_mean_since_beginning_window_2',
    'basket_source_y_mean_since_beginning_window_2',
    'basket_source_z_mean_since_beginning_window_2',
    'basket_source_roll_mean_since_beginning_window_2',
    'basket_source_pitch_mean_since_beginning_window_2',
    'basket_source_yaw_mean_since_beginning_window_2',
    'fryer_slot_x_mean_since_beginning_window_2',
    'fryer_slot_y_mean_since_beginning_window_2',
    'fryer_slot_z_mean_since_beginning_window_2',
    'fryer_slot_roll_mean_since_beginning_window_2',
    'fryer_slot_pitch_mean_since_beginning_window_2',
    'fryer_slot_yaw_mean_since_beginning_window_2',
    'hanger_tool_slot_x_mean_since_beginning_window_2',
    'hanger_tool_slot_y_mean_since_beginning_window_2',
    'hanger_tool_slot_z_mean_since_beginning_window_2',
    'hanger_tool_slot_roll_mean_since_beginning_window_2',
    'hanger_tool_slot_pitch_mean_since_beginning_window_2',
    'hanger_tool_slot_yaw_mean_since_beginning_window_2',
    'basket_source_no_offset_x_mean_since_beginning_window_2',
    'basket_source_no_offset_y_mean_since_beginning_window_2',
    'basket_source_no_offset_z_mean_since_beginning_window_2',
    'basket_source_no_offset_roll_mean_since_beginning_window_2',
    'basket_source_no_offset_pitch_mean_since_beginning_window_2',
    'basket_source_no_offset_yaw_mean_since_beginning_window_2',
    'fryer_slot_no_offset_x_mean_since_beginning_window_2',
    'fryer_slot_no_offset_y_mean_since_beginning_window_2',
    'fryer_slot_no_offset_z_mean_since_beginning_window_2',
    'fryer_slot_no_offset_roll_mean_since_beginning_window_2',
    'fryer_slot_no_offset_pitch_mean_since_beginning_window_2',
    'fryer_slot_no_offset_yaw_mean_since_beginning_window_2',
    'hanger_tool_slot_no_offset_x_mean_since_beginning_window_2',
    'hanger_tool_slot_no_offset_y_mean_since_beginning_window_2',
    'hanger_tool_slot_no_offset_z_mean_since_beginning_window_2',
    'hanger_tool_slot_no_offset_roll_mean_since_beginning_window_2',
    'hanger_tool_slot_no_offset_pitch_mean_since_beginning_window_2',
    'hanger_tool_slot_no_offset_yaw_mean_since_beginning_window_2',
    'fryer_target_abs_offset_x_mean_since_beginning_window_2',
    'fryer_target_abs_offset_y_mean_since_beginning_window_2',
    'fryer_target_abs_offset_z_mean_since_beginning_window_2',
    'fryer_target_rel_offset_x_mean_since_beginning_window_2',
    'fryer_target_rel_offset_y_mean_since_beginning_window_2',
    'fryer_target_rel_offset_z_mean_since_beginning_window_2',
    'hanger_target_abs_offset_x_mean_since_beginning_window_2',
    'hanger_target_abs_offset_y_mean_since_beginning_window_2',
    'hanger_target_abs_offset_z_mean_since_beginning_window_2',
    'hanger_target_rel_offset_x_mean_since_beginning_window_2',
    'hanger_target_rel_offset_y_mean_since_beginning_window_2',
    'hanger_target_rel_offset_z_mean_since_beginning_window_2',
    'basket_source_abs_offset_x_mean_since_beginning_window_2',
    'basket_source_abs_offset_y_mean_since_beginning_window_2',
    'basket_source_abs_offset_z_mean_since_beginning_window_2',
    'basket_source_rel_offset_x_mean_since_beginning_window_2',
    'basket_source_rel_offset_y_mean_since_beginning_window_2',
    'basket_source_rel_offset_z_mean_since_beginning_window_2',
    'error_code_0.0_mean_since_beginning_window_11',
    'error_code_1.0_mean_since_beginning_window_11',
    'basket_source_x_mean_since_beginning_window_11',
    'basket_source_y_mean_since_beginning_window_11',
    'basket_source_z_mean_since_beginning_window_11',
    'basket_source_roll_mean_since_beginning_window_11',
    'basket_source_pitch_mean_since_beginning_window_11',
    'basket_source_yaw_mean_since_beginning_window_11',
    'fryer_slot_x_mean_since_beginning_window_11',
    'fryer_slot_y_mean_since_beginning_window_11',
    'fryer_slot_z_mean_since_beginning_window_11',
    'fryer_slot_roll_mean_since_beginning_window_11',
    'fryer_slot_pitch_mean_since_beginning_window_11',
    'fryer_slot_yaw_mean_since_beginning_window_11',
    'hanger_tool_slot_x_mean_since_beginning_window_11',
    'hanger_tool_slot_y_mean_since_beginning_window_11',
    'hanger_tool_slot_z_mean_since_beginning_window_11',
    'hanger_tool_slot_roll_mean_since_beginning_window_11',
    'hanger_tool_slot_pitch_mean_since_beginning_window_11',
    'hanger_tool_slot_yaw_mean_since_beginning_window_11',
    'basket_source_no_offset_x_mean_since_beginning_window_11',
    'basket_source_no_offset_y_mean_since_beginning_window_11',
    'basket_source_no_offset_z_mean_since_beginning_window_11',
    'basket_source_no_offset_roll_mean_since_beginning_window_11',
    'basket_source_no_offset_pitch_mean_since_beginning_window_11',
    'basket_source_no_offset_yaw_mean_since_beginning_window_11',
    'fryer_slot_no_offset_x_mean_since_beginning_window_11',
    'fryer_slot_no_offset_y_mean_since_beginning_window_11',
    'fryer_slot_no_offset_z_mean_since_beginning_window_11',
    'fryer_slot_no_offset_roll_mean_since_beginning_window_11',
    'fryer_slot_no_offset_pitch_mean_since_beginning_window_11',
    'fryer_slot_no_offset_yaw_mean_since_beginning_window_11',
    'hanger_tool_slot_no_offset_x_mean_since_beginning_window_11',
    'hanger_tool_slot_no_offset_y_mean_since_beginning_window_11',
    'hanger_tool_slot_no_offset_z_mean_since_beginning_window_11',
    'hanger_tool_slot_no_offset_roll_mean_since_beginning_window_11',
    'hanger_tool_slot_no_offset_pitch_mean_since_beginning_window_11',
    'hanger_tool_slot_no_offset_yaw_mean_since_beginning_window_11',
    'fryer_target_abs_offset_x_mean_since_beginning_window_11',
    'fryer_target_abs_offset_y_mean_since_beginning_window_11',
    'fryer_target_abs_offset_z_mean_since_beginning_window_11',
    'fryer_target_rel_offset_x_mean_since_beginning_window_11',
    'fryer_target_rel_offset_y_mean_since_beginning_window_11',
    'fryer_target_rel_offset_z_mean_since_beginning_window_11',
    'hanger_target_abs_offset_x_mean_since_beginning_window_11',
    'hanger_target_abs_offset_y_mean_since_beginning_window_11',
    'hanger_target_abs_offset_z_mean_since_beginning_window_11',
    'hanger_target_rel_offset_x_mean_since_beginning_window_11',
    'hanger_target_rel_offset_y_mean_since_beginning_window_11',
    'hanger_target_rel_offset_z_mean_since_beginning_window_11',
    'basket_source_abs_offset_x_mean_since_beginning_window_11',
    'basket_source_abs_offset_y_mean_since_beginning_window_11',
    'basket_source_abs_offset_z_mean_since_beginning_window_11',
    'basket_source_rel_offset_x_mean_since_beginning_window_11',
    'basket_source_rel_offset_y_mean_since_beginning_window_11',
    'basket_source_rel_offset_z_mean_since_beginning_window_11',
    'error_code_0.0_mean_since_beginning_window_51',
    'error_code_1.0_mean_since_beginning_window_51',
    'basket_source_x_mean_since_beginning_window_51',
    'basket_source_y_mean_since_beginning_window_51',
    'basket_source_z_mean_since_beginning_window_51',
    'basket_source_roll_mean_since_beginning_window_51',
    'basket_source_pitch_mean_since_beginning_window_51',
    'basket_source_yaw_mean_since_beginning_window_51',
    'fryer_slot_x_mean_since_beginning_window_51',
    'fryer_slot_y_mean_since_beginning_window_51',
    'fryer_slot_z_mean_since_beginning_window_51',
    'fryer_slot_roll_mean_since_beginning_window_51',
    'fryer_slot_pitch_mean_since_beginning_window_51',
    'fryer_slot_yaw_mean_since_beginning_window_51',
    'hanger_tool_slot_x_mean_since_beginning_window_51',
    'hanger_tool_slot_y_mean_since_beginning_window_51',
    'hanger_tool_slot_z_mean_since_beginning_window_51',
    'hanger_tool_slot_roll_mean_since_beginning_window_51',
    'hanger_tool_slot_pitch_mean_since_beginning_window_51',
    'hanger_tool_slot_yaw_mean_since_beginning_window_51',
    'basket_source_no_offset_x_mean_since_beginning_window_51',
    'basket_source_no_offset_y_mean_since_beginning_window_51',
    'basket_source_no_offset_z_mean_since_beginning_window_51',
    'basket_source_no_offset_roll_mean_since_beginning_window_51',
    'basket_source_no_offset_pitch_mean_since_beginning_window_51',
    'basket_source_no_offset_yaw_mean_since_beginning_window_51',
    'fryer_slot_no_offset_x_mean_since_beginning_window_51',
    'fryer_slot_no_offset_y_mean_since_beginning_window_51',
    'fryer_slot_no_offset_z_mean_since_beginning_window_51',
    'fryer_slot_no_offset_roll_mean_since_beginning_window_51',
    'fryer_slot_no_offset_pitch_mean_since_beginning_window_51',
    'fryer_slot_no_offset_yaw_mean_since_beginning_window_51',
    'hanger_tool_slot_no_offset_x_mean_since_beginning_window_51',
    'hanger_tool_slot_no_offset_y_mean_since_beginning_window_51',
    'hanger_tool_slot_no_offset_z_mean_since_beginning_window_51',
    'hanger_tool_slot_no_offset_roll_mean_since_beginning_window_51',
    'hanger_tool_slot_no_offset_pitch_mean_since_beginning_window_51',
    'hanger_tool_slot_no_offset_yaw_mean_since_beginning_window_51',
    'fryer_target_abs_offset_x_mean_since_beginning_window_51',
    'fryer_target_abs_offset_y_mean_since_beginning_window_51',
    'fryer_target_abs_offset_z_mean_since_beginning_window_51',
    'fryer_target_rel_offset_x_mean_since_beginning_window_51',
    'fryer_target_rel_offset_y_mean_since_beginning_window_51',
    'fryer_target_rel_offset_z_mean_since_beginning_window_51',
    'hanger_target_abs_offset_x_mean_since_beginning_window_51',
    'hanger_target_abs_offset_y_mean_since_beginning_window_51',
    'hanger_target_abs_offset_z_mean_since_beginning_window_51',
    'hanger_target_rel_offset_x_mean_since_beginning_window_51',
    'hanger_target_rel_offset_y_mean_since_beginning_window_51',
    'hanger_target_rel_offset_z_mean_since_beginning_window_51',
    'basket_source_abs_offset_x_mean_since_beginning_window_51',
    'basket_source_abs_offset_y_mean_since_beginning_window_51',
    'basket_source_abs_offset_z_mean_since_beginning_window_51',
    'basket_source_rel_offset_x_mean_since_beginning_window_51',
    'basket_source_rel_offset_y_mean_since_beginning_window_51',
    'basket_source_rel_offset_z_mean_since_beginning_window_51',
    'error_code_0.0_mean_since_beginning_window_101',
    'error_code_1.0_mean_since_beginning_window_101',
    'basket_source_x_mean_since_beginning_window_101',
    'basket_source_y_mean_since_beginning_window_101',
    'basket_source_z_mean_since_beginning_window_101',
    'basket_source_roll_mean_since_beginning_window_101',
    'basket_source_pitch_mean_since_beginning_window_101',
    'basket_source_yaw_mean_since_beginning_window_101',
    'fryer_slot_x_mean_since_beginning_window_101',
    'fryer_slot_y_mean_since_beginning_window_101',
    'fryer_slot_z_mean_since_beginning_window_101',
    'fryer_slot_roll_mean_since_beginning_window_101',
    'fryer_slot_pitch_mean_since_beginning_window_101',
    'fryer_slot_yaw_mean_since_beginning_window_101',
    'hanger_tool_slot_x_mean_since_beginning_window_101',
    'hanger_tool_slot_y_mean_since_beginning_window_101',
    'hanger_tool_slot_z_mean_since_beginning_window_101',
    'hanger_tool_slot_roll_mean_since_beginning_window_101',
    'hanger_tool_slot_pitch_mean_since_beginning_window_101',
    'hanger_tool_slot_yaw_mean_since_beginning_window_101',
    'basket_source_no_offset_x_mean_since_beginning_window_101',
    'basket_source_no_offset_y_mean_since_beginning_window_101',
    'basket_source_no_offset_z_mean_since_beginning_window_101',
    'basket_source_no_offset_roll_mean_since_beginning_window_101',
    'basket_source_no_offset_pitch_mean_since_beginning_window_101',
    'basket_source_no_offset_yaw_mean_since_beginning_window_101',
    'fryer_slot_no_offset_x_mean_since_beginning_window_101',
    'fryer_slot_no_offset_y_mean_since_beginning_window_101',
    'fryer_slot_no_offset_z_mean_since_beginning_window_101',
    'fryer_slot_no_offset_roll_mean_since_beginning_window_101',
    'fryer_slot_no_offset_pitch_mean_since_beginning_window_101',
    'fryer_slot_no_offset_yaw_mean_since_beginning_window_101',
    'hanger_tool_slot_no_offset_x_mean_since_beginning_window_101',
    'hanger_tool_slot_no_offset_y_mean_since_beginning_window_101',
    'hanger_tool_slot_no_offset_z_mean_since_beginning_window_101',
    'hanger_tool_slot_no_offset_roll_mean_since_beginning_window_101',
    'hanger_tool_slot_no_offset_pitch_mean_since_beginning_window_101',
    'hanger_tool_slot_no_offset_yaw_mean_since_beginning_window_101',
    'fryer_target_abs_offset_x_mean_since_beginning_window_101',
    'fryer_target_abs_offset_y_mean_since_beginning_window_101',
    'fryer_target_abs_offset_z_mean_since_beginning_window_101',
    'fryer_target_rel_offset_x_mean_since_beginning_window_101',
    'fryer_target_rel_offset_y_mean_since_beginning_window_101',
    'fryer_target_rel_offset_z_mean_since_beginning_window_101',
    'hanger_target_abs_offset_x_mean_since_beginning_window_101',
    'hanger_target_abs_offset_y_mean_since_beginning_window_101',
    'hanger_target_abs_offset_z_mean_since_beginning_window_101',
    'hanger_target_rel_offset_x_mean_since_beginning_window_101',
    'hanger_target_rel_offset_y_mean_since_beginning_window_101',
    'hanger_target_rel_offset_z_mean_since_beginning_window_101',
    'basket_source_abs_offset_x_mean_since_beginning_window_101',
    'basket_source_abs_offset_y_mean_since_beginning_window_101',
    'basket_source_abs_offset_z_mean_since_beginning_window_101',
    'basket_source_rel_offset_x_mean_since_beginning_window_101',
    'basket_source_rel_offset_y_mean_since_beginning_window_101',
    'basket_source_rel_offset_z_mean_since_beginning_window_101',
    'error_code_0.0_mean_since_beginning_window_250',
    'error_code_1.0_mean_since_beginning_window_250',
    'basket_source_x_mean_since_beginning_window_250',
    'basket_source_y_mean_since_beginning_window_250',
    'basket_source_z_mean_since_beginning_window_250',
    'basket_source_roll_mean_since_beginning_window_250',
    'basket_source_pitch_mean_since_beginning_window_250',
    'basket_source_yaw_mean_since_beginning_window_250',
    'fryer_slot_x_mean_since_beginning_window_250',
    'fryer_slot_y_mean_since_beginning_window_250',
    'fryer_slot_z_mean_since_beginning_window_250',
    'fryer_slot_roll_mean_since_beginning_window_250',
    'fryer_slot_pitch_mean_since_beginning_window_250',
    'fryer_slot_yaw_mean_since_beginning_window_250',
    'hanger_tool_slot_x_mean_since_beginning_window_250',
    'hanger_tool_slot_y_mean_since_beginning_window_250',
    'hanger_tool_slot_z_mean_since_beginning_window_250',
    'hanger_tool_slot_roll_mean_since_beginning_window_250',
    'hanger_tool_slot_pitch_mean_since_beginning_window_250',
    'hanger_tool_slot_yaw_mean_since_beginning_window_250',
    'basket_source_no_offset_x_mean_since_beginning_window_250',
    'basket_source_no_offset_y_mean_since_beginning_window_250',
    'basket_source_no_offset_z_mean_since_beginning_window_250',
    'basket_source_no_offset_roll_mean_since_beginning_window_250',
    'basket_source_no_offset_pitch_mean_since_beginning_window_250',
    'basket_source_no_offset_yaw_mean_since_beginning_window_250',
    'fryer_slot_no_offset_x_mean_since_beginning_window_250',
    'fryer_slot_no_offset_y_mean_since_beginning_window_250',
    'fryer_slot_no_offset_z_mean_since_beginning_window_250',
    'fryer_slot_no_offset_roll_mean_since_beginning_window_250',
    'fryer_slot_no_offset_pitch_mean_since_beginning_window_250',
    'fryer_slot_no_offset_yaw_mean_since_beginning_window_250',
    'hanger_tool_slot_no_offset_x_mean_since_beginning_window_250',
    'hanger_tool_slot_no_offset_y_mean_since_beginning_window_250',
    'hanger_tool_slot_no_offset_z_mean_since_beginning_window_250',
    'hanger_tool_slot_no_offset_roll_mean_since_beginning_window_250',
    'hanger_tool_slot_no_offset_pitch_mean_since_beginning_window_250',
    'hanger_tool_slot_no_offset_yaw_mean_since_beginning_window_250',
    'fryer_target_abs_offset_x_mean_since_beginning_window_250',
    'fryer_target_abs_offset_y_mean_since_beginning_window_250',
    'fryer_target_abs_offset_z_mean_since_beginning_window_250',
    'fryer_target_rel_offset_x_mean_since_beginning_window_250',
    'fryer_target_rel_offset_y_mean_since_beginning_window_250',
    'fryer_target_rel_offset_z_mean_since_beginning_window_250',
    'hanger_target_abs_offset_x_mean_since_beginning_window_250',
    'hanger_target_abs_offset_y_mean_since_beginning_window_250',
    'hanger_target_abs_offset_z_mean_since_beginning_window_250',
    'hanger_target_rel_offset_x_mean_since_beginning_window_250',
    'hanger_target_rel_offset_y_mean_since_beginning_window_250',
    'hanger_target_rel_offset_z_mean_since_beginning_window_250',
    'basket_source_abs_offset_x_mean_since_beginning_window_250',
    'basket_source_abs_offset_y_mean_since_beginning_window_250',
    'basket_source_abs_offset_z_mean_since_beginning_window_250',
    'basket_source_rel_offset_x_mean_since_beginning_window_250',
    'basket_source_rel_offset_y_mean_since_beginning_window_250',
    'basket_source_rel_offset_z_mean_since_beginning_window_250',
    'error_code_0.0_mean_since_beginning_window_1001',
    'error_code_1.0_mean_since_beginning_window_1001',
    'basket_source_x_mean_since_beginning_window_1001',
    'basket_source_y_mean_since_beginning_window_1001',
    'basket_source_z_mean_since_beginning_window_1001',
    'basket_source_roll_mean_since_beginning_window_1001',
    'basket_source_pitch_mean_since_beginning_window_1001',
    'basket_source_yaw_mean_since_beginning_window_1001',
    'fryer_slot_x_mean_since_beginning_window_1001',
    'fryer_slot_y_mean_since_beginning_window_1001',
    'fryer_slot_z_mean_since_beginning_window_1001',
    'fryer_slot_roll_mean_since_beginning_window_1001',
    'fryer_slot_pitch_mean_since_beginning_window_1001',
    'fryer_slot_yaw_mean_since_beginning_window_1001',
    'hanger_tool_slot_x_mean_since_beginning_window_1001',
    'hanger_tool_slot_y_mean_since_beginning_window_1001',
    'hanger_tool_slot_z_mean_since_beginning_window_1001',
    'hanger_tool_slot_roll_mean_since_beginning_window_1001',
    'hanger_tool_slot_pitch_mean_since_beginning_window_1001',
    'hanger_tool_slot_yaw_mean_since_beginning_window_1001',
    'basket_source_no_offset_x_mean_since_beginning_window_1001',
    'basket_source_no_offset_y_mean_since_beginning_window_1001',
    'basket_source_no_offset_z_mean_since_beginning_window_1001',
    'basket_source_no_offset_roll_mean_since_beginning_window_1001',
    'basket_source_no_offset_pitch_mean_since_beginning_window_1001',
    'basket_source_no_offset_yaw_mean_since_beginning_window_1001',
    'fryer_slot_no_offset_x_mean_since_beginning_window_1001',
    'fryer_slot_no_offset_y_mean_since_beginning_window_1001',
    'fryer_slot_no_offset_z_mean_since_beginning_window_1001',
    'fryer_slot_no_offset_roll_mean_since_beginning_window_1001',
    'fryer_slot_no_offset_pitch_mean_since_beginning_window_1001',
    'fryer_slot_no_offset_yaw_mean_since_beginning_window_1001',
    'hanger_tool_slot_no_offset_x_mean_since_beginning_window_1001',
    'hanger_tool_slot_no_offset_y_mean_since_beginning_window_1001',
    'hanger_tool_slot_no_offset_z_mean_since_beginning_window_1001',
    'hanger_tool_slot_no_offset_roll_mean_since_beginning_window_1001',
    'hanger_tool_slot_no_offset_pitch_mean_since_beginning_window_1001',
    'hanger_tool_slot_no_offset_yaw_mean_since_beginning_window_1001',
    'fryer_target_abs_offset_x_mean_since_beginning_window_1001',
    'fryer_target_abs_offset_y_mean_since_beginning_window_1001',
    'fryer_target_abs_offset_z_mean_since_beginning_window_1001',
    'fryer_target_rel_offset_x_mean_since_beginning_window_1001',
    'fryer_target_rel_offset_y_mean_since_beginning_window_1001',
    'fryer_target_rel_offset_z_mean_since_beginning_window_1001',
    'hanger_target_abs_offset_x_mean_since_beginning_window_1001',
    'hanger_target_abs_offset_y_mean_since_beginning_window_1001',
    'hanger_target_abs_offset_z_mean_since_beginning_window_1001',
    'hanger_target_rel_offset_x_mean_since_beginning_window_1001',
    'hanger_target_rel_offset_y_mean_since_beginning_window_1001',
    'hanger_target_rel_offset_z_mean_since_beginning_window_1001',
    'basket_source_abs_offset_x_mean_since_beginning_window_1001',
    'basket_source_abs_offset_y_mean_since_beginning_window_1001',
    'basket_source_abs_offset_z_mean_since_beginning_window_1001',
    'basket_source_rel_offset_x_mean_since_beginning_window_1001',
    'basket_source_rel_offset_y_mean_since_beginning_window_1001',
    'basket_source_rel_offset_z_mean_since_beginning_window_1001',
]

# print(combined_df.head(1))

print(len(feature_cols), len(combined_df.columns.to_list()))


input_data = combined_df[feature_cols]

print(len(input_data.columns))

# Convert DataFrame to numpy array
# input_data = input_data.to_numpy()