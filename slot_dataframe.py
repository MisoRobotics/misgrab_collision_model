    
import pandas as pd
import numpy as np

class SlotDataFrame:
    def __init__(main_df, fryer_slot_id, slot_df_len_limit) -> None:
        """instantiates slot_df from main df along with row lim, lim"""
        self.slot_df_len_limit = slot_df_len_limit
        self.slot_df = main_df[main_df['fryer_slot_id'] == fryer_slot_id].tail(slot_df_len_limit)

    def get_len(self) -> int:
        """returns length of slot_df"""
        return len(self.slot_df)
    
    def pop_row(self) -> None:
        """pops off the oldest row in the slot dataframe"""
        oldest_index = self.slot_df.index[0]
        shortened_df = self.slot_df.drop(oldest_index)
        del(self.slot_df)
        self.slot_df = shortened_df

    def add_row(self, incoming_row) -> None:
        """takes in incoming row in the form of a list, pops off oldest row in df if necessary
        adds latest row to slot_df"""
        if self.get_len >= self.slot_df_len_limit:
            self.pop_row()
        self.slot_df = pd.DataFrame([incoming_row], columns=self.slot_df.columns).append(self.slot_df, ignore_index=True)
    
    def get_model_input(self, incoming_localization_data) -> np.array:
        """takes in the incoming localization data, adds the other feature columns to it 
        including average based ones, returns numpy array to be used as input to pred model"""
        pass
        #cols in incoming_localizaton_data:
        #same as the columns in the slot_df with the exception of whether the execution was successful or not / error codes/ stuff we can't
        #know until it happens 
