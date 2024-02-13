
# 1. at flippy start up, one dataframe is created for each fryer slot dating back to more than 1000 rows (probably like 2000) with and without offsets
# 2. if a new row is added to the behaviors table OR to the offset atlas table then we update the appropriate fryer slot dataframe AND we delete the last row from that dataframe (if it is already longer than 2000ish rows)
# 3. if a behavior is starting to be planned, then we determine the apprioriate dataframe to use for reference
# 4. we obtain the weighted averages and other necessary features from that df
# 5. we create an inference from those features

# FOR PREDICTIONS:
# - we need a way to disable predictions or ignore them
#     -- for instance if something is still loading like in initialization
#     -- if this is the first occurence of this behavior with this slot and basket so we don't have other data
#     -- if we are having too many false positives
#     -- manually disable predictions
#     -- for AB testing
#NEED way to keep track of whether a mitigation measure was just undertaken, such as 
#whether requerying was already attempted or not so we know whether to requery or not, I am
#thinking that adding in extra cols to the dataframe could be helpful for this

import pandas as pd
import tensorflow as tf
import numpy as np
from slot_dataframe import SlotDataFrame
from predictor import Predictor
from utils import read_yaml_to_dict


#TODO: add in yaml config file for the lists of columns , also yaml utility
#TODO: yaml file will contain the column name, whether it has any dummies assoc with it, whether it
#TODO: is part of the features, whether it is the target, whether it has any rolling averages assoc with it
#TODO: if it does, then what is the averaging window and what column is the rolling average based off of
#TODO: this yaml file will be used to perform the feature engineering for the input
#TODO: There may be another yaml file for the slot dataframes, along with for the main dataframe
#TODO: for the slot_dfs the yaml file would include info about which extra cols to add etc


CSV_PATH = "/home/sam/Downloads/behaviors_df.csv"
MODEL_PATH = "/home/sam/Downloads/misgrab_prediction/models/test_rename"
EXPECTED_COLUMNS = []
ENGINEER_FEATURES_CONFIG_FILE_PATH = "/home/sam/Downloads/misgrab_prediction/package_model2/engineer_features_config.yaml"

df = pd.read_csv(CSV_PATH)



class MainDataFrame:
    def __init__(self, model_path, expected_columns, engineer_features_config_file_path)-> None:
        """loads main_df (from query or from mock db in the form of csv),
        creates all slot dataframe instances, stores in dict"""
        #insert initialization for the main df (query the behaviors table, set up slot dataframes)
        self.main_df = pd.read_csv(CSV_PATH)
        self.engineer_features_config_file_path = engineer_features_config_file_path
        self.engineer_features_config_dict = read_yaml_to_dict(engineer_features_config_file_path)
        fryer_slot_id_list = [0,1,2,3,4,5,6,7] #include 1-8 not 0-7
        self.fryer_df_dict = dict()
        for fryer_slot_id in fryer_slot_id_list:
            self.fryer_df_dict[fryer_slot_id] = SlotDataFrame(self.main_df, fryer_slot_id, lim=2000)
        self.predictor = Predictor(model_path, expected_columns)
    
    def receive_incoming_localization_data(self, incoming_localization_data) -> bool:
        """takes incoming_localizatoin_data, determines slot id, returns output of get_model_input
        for appropriate slot_df (the prediction)"""
        #assume that incoming_localization_data is a dataframe consisting of one row
        matching_slot_df = self.fryer_df_dict[incoming_localization_data['fryer_slot_id']]
        pred_input_data = matching_slot_df.get_model_input(incoming_localization_data)
        return self.predictor.make_pred(pred_input_data)
    
    def receive_incoming_new_row(self, incoming_new_row) -> None:
        """takes incoming new row, determines slot id, calls add_row method for appropriate
        slot_df"""
        #assume that incoming_new_row is a dataframe consisting of one row
        matching_slot_df = self.fryer_df_dict[incoming_new_row['fryer_slot_id']]
        matching_slot_df.add_row(incoming_new_row)

