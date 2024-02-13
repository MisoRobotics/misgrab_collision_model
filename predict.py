from helpers import *
import pandas as pd

df = pd.read_csv("../../behaviors_df.csv")



import pandas as pd
import tensorflow as tf

# Assuming model_path is correctly pointing to the directory containing the SavedModel
model_path = "/home/sam/Downloads/misgrab_prediction/models/test_rename"
csv_path = "/home/sam/Downloads/misgrab_prediction/int_32_new_added_features_1-30.csv"
# csv_path = "./sampled_output.csv"
def float_32_df_import(file_path):
    df_preview = pd.read_csv(file_path, nrows=0)
    dtype_dict = {col: 'float32' for col in df_preview.columns}

    object_columns = ['site_id', 'behavior_start_time', 'basket_state', 'Failure Description']
    for col in object_columns:
        dtype_dict[col] = 'object'

    df2 = pd.read_csv(file_path, dtype=dtype_dict)
    return df2

print('loading df')
# df = float_32_df_import(csv_path)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

feature_columns_to_exclude = [
    'site_id', 'behavior_start_time', 'basket_id', 'error_code', 'basket_state',
    'fryer_slot_id', 'Failure Description', 'error_code_0.0', 'error_code_1.0'
]

df2 = df.drop(columns=feature_columns_to_exclude)


rand_rows = []
for _ in range(100):
    rand_row = df.sample(n=1).drop(columns=feature_columns_to_exclude)
    rand_rows.append(rand_row)

df['error_code_0.0'] = df['Failure Description'] == 

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
 
#  'fryer_target_abs_offset_x',
# 'fryer_target_abs_offset_y',
# 'fryer_target_abs_offset_z',
# 'fryer_target_rel_offset_x',
# 'fryer_target_rel_offset_y',
# 'fryer_target_rel_offset_z',
# 'hanger_target_abs_offset_x',
# 'hanger_target_abs_offset_y',
# 'hanger_target_abs_offset_z',
# 'hanger_target_rel_offset_x',
# 'hanger_target_rel_offset_y',
# 'hanger_target_rel_offset_z',
# 'basket_source_abs_offset_x',
# 'basket_source_abs_offset_y',
# 'basket_source_abs_offset_z',
# 'basket_source_rel_offset_x',
# 'basket_source_rel_offset_y',
# 'basket_source_rel_offset_z',
]


df3 = add_moving_averages(df, ec_col, [1,2,10,250,1000])

for col in df3.columns:
    print(col)

