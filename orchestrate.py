import pandas as pd
from slot_dataframe import SlotDataFrame
from predictor import Predictor
from unoptimized_query2 import get_combined_df

class MainDataFrame:
    def __init__(self, config_dict)-> None:
        """loads main_df (from query or from mock db in the form of csv),
        creates all slot dataframe instances, stores in dict"""
        #insert initialization for the main df (query the behaviors table, set up slot dataframes)
        self.config_dict = config_dict

        if config_dict['LOAD_CSV']:
            self.main_df = pd.read_csv(config_dict['CSV_PATH'])
        else:
            self.main_df = get_combined_df(start_time=initial_start)

        fryer_slot_id_list = [1,2,3,4,5,6,7,8]
        self.fryer_df_dict = dict()
        for fryer_slot_id in fryer_slot_id_list:
            self.fryer_df_dict[fryer_slot_id] = SlotDataFrame(self.main_df, fryer_slot_id, self.config_dict)
        
        self.predictor = Predictor(config_dict['MODEL_PATH'], config_dict["FEATURE_COLS"])
    
    def receive_incoming_localization_data(self, incoming_localization_data) -> bool:
        """takes incoming_localizatoin_data, determines slot id, returns output of get_model_input
        for appropriate slot_df (the prediction)"""
        #assume that incoming_localization_data is a dataframe consisting of one row
        matching_slot_df = self.fryer_df_dict[incoming_localization_data.iloc[0]['fryer_slot_id']]
        pred_input_data = matching_slot_df.get_model_input(incoming_localization_data)
        return self.predictor.make_pred(pred_input_data)
    
    def receive_incoming_new_row(self, incoming_new_row) -> None:
        """takes incoming new row, determines slot id, calls add_row method for appropriate
        slot_df"""
        #assume that incoming_new_row is a dataframe consisting of one row
        matching_slot_df = self.fryer_df_dict[incoming_new_row['fryer_slot_id']]
        matching_slot_df.add_row(incoming_new_row)

