    
import pandas as pd
import numpy as np
import utils


class SlotDataFrame:
    def __init__(self, main_df: pd.DataFrame, fryer_slot_id: int, config_dict: dict) -> None:
        """instantiates slot_df from main df along with row lim, lim"""
        self.config_dict = config_dict
        self.slot_df_len_limit = config_dict["SLOT_DF_LEN_LIMIT"]
        self.slot_df = utils.create_slot_df(main_df, self.config_dict["MISGRABS_OR_COLLISIONS"], self.slot_df_len_limit, fryer_slot_id, self.config_dict["FEATURE_DUMMY_DICT"])
        self.fryer_slot_id = fryer_slot_id

    def get_len(self) -> int:
        """returns length of slot_df"""
        return len(self.slot_df)
    
    def pop_row(self) -> None:
        """pops off the oldest row in the slot dataframe"""
        oldest_index = self.slot_df.index[0]
        shortened_df = self.slot_df.drop(oldest_index)
        del(self.slot_df)
        self.slot_df = shortened_df

    def add_row(self, incoming_row: pd.DataFrame) -> None:
        """takes in incoming row in the form of a list, pops off oldest row in df if necessary
        adds latest row to slot_df"""
        if self.get_len >= self.slot_df_len_limit:
            self.pop_row()
        for source_col, categorical_cols in self.config_dict['FEATURE_DUMMY_DICT'].items():
            utils.add_categoricals(incoming_row, categorical_cols, source_col)
        self.slot_df = pd.DataFrame([incoming_row], columns=self.slot_df.columns).append(self.slot_df, ignore_index=True)
    

    def get_model_input(self, incoming_localization_data: pd.DataFrame) -> pd.DataFrame:
        """takes in the incoming localization data, adds the other feature columns to it 
        including average based ones, returns one row df to be used as input to pred model"""
        new_slot_df = utils.rename_columns(self.slot_df, self.config_dict["COLS_TO_RENAME"], self.config_dict["COLS_TO_RENAME_TO"])
        combined_input_and_slot_avgs_df = utils.prepare_input_data(new_slot_df, incoming_localization_data, self.config_dict["AVG_WINDOWS"], self.config_dict["COL_TO_AVG"],
                                                                    int(incoming_localization_data.iloc[0]["basket_id"]), int(incoming_localization_data.iloc[0]["fryer_slot_id"]), 
                                                                   str(incoming_localization_data.iloc[0]["basket_state"]), self.config_dict["FEATURE_DUMMY_DICT"],
                                                                   self.config_dict["COLS_TO_RENAME"], self.config_dict["COLS_TO_RENAME_TO"], 
                                                                   self.config_dict["COL_TO_DROP_EARLY"], self.config_dict["SITE_ID"], self.config_dict["SITE_ID_POSSIBILITIES"])

        return combined_input_and_slot_avgs_df