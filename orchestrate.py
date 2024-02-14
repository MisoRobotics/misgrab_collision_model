import pandas as pd
from slot_dataframe import SlotDataFrame
from predictor import Predictor
from query import get_combined_data 
import datetime
from utils import convert_str_time, create_slot_df, get_current_time
import asyncio

class MainDataFrame:
    def __init__(self, config_dict: dict)-> None:
        """loads main_df (from query or from mock db in the form of csv),
        creates all slot dataframe instances, stores in dict"""
        #insert initialization for the main df (query the behaviors table, set up slot dataframes)
        self.config_dict = config_dict

        if config_dict['LOAD_CSV']:
            self.main_df = pd.read_csv(config_dict['CSV_PATH'])
        else:
            self.main_df = get_combined_data(
                self.config_dict["BEHAVIORS_INIT_START_TIME"],
                self.config_dict["BEHAVIORS_INIT_END_TIME"],
                self.config_dict["OFFSETS_INIT_START_TIME"],
                self.config_dict["OFFSETS_INIT_END_TIME"],
            )
            self.last_time = self.config_dict["BEHAVIORS_INIT_END_TIME"]
    
        self.fryer_slot_id_list = [1, 2, 3, 4, 5, 6, 7, 8]
        self.slot_dfs = dict()
        for fryer_slot_id in self.fryer_slot_id_list:
            self.slot_dfs[fryer_slot_id] = SlotDataFrame(self.main_df, fryer_slot_id, self.config_dict)

        self.predictor = Predictor(config_dict['MODEL_PATH'], config_dict["FEATURE_COLS"])
        
        # self.loop = asyncio.get_event_loop()  # Initialize the event loop reference
        # self.background_polling = self.loop.create_task(self.poll_for_updates())
        self.update_slot_dfs()


    def get_slot_df_from_id(self, fryer_slot_id: int) -> pd.DataFrame:
        return self.slot_dfs[fryer_slot_id].slot_df

    def get_n_random_rows(self, num):
        return [self.main_df.sample(n=1) for _ in range(num)]
    def update_fryer_slot_df_by_id(self, fryer_slot_id, df_to_replace_with):
        self.slot_dfs[fryer_slot_id].slot_df = df_to_replace_with

    def update_slot_dfs(self):
        print("UPDATER WAS CALLED!!\n\n")
        now = get_current_time()
        # now = convert_str_time(datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
        new_rows = get_combined_data(self.last_time, now, self.last_time, now)
        for fryer_slot_id in self.fryer_slot_id_list:
            new_rows_for_slot_df = new_rows[new_rows['fryer_slot_id'] == fryer_slot_id]
            if len(new_rows_for_slot_df) == 0:
                print('length was zero')
                continue
            curr_slot_df = self.get_slot_df_from_id(fryer_slot_id)
            print(f"len of curr_slot_df before: {len(curr_slot_df)}")
            new_rows_for_slot_df_filtered = create_slot_df(
                            new_rows_for_slot_df,
                            self.config_dict["MISGRABS_OR_COLLISIONS"],
                            self.config_dict["SLOT_DF_LEN_LIMIT"], 
                            fryer_slot_id, 
                            self.config_dict["FEATURE_DUMMY_DICT"]
                            )
            
            combined_df = pd.concat([curr_slot_df, new_rows_for_slot_df_filtered]).sort_values(by="behavior_start_time", ascending=True)
            self.update_fryer_slot_df_by_id(fryer_slot_id, combined_df)
            print(f"len of curr_slot_df after: {len(self.get_slot_df_from_id(fryer_slot_id))}")

        self.last_time = now

    # async def poll_for_updates(self):
    #     while True:
    #         # Your background task logic here
    #         print("Background task is running...")
    #         self.update_slot_dfs()
    #         await asyncio.sleep(10)  # Non-blocking sleep

    def receive_incoming_localization_data(self, incoming_localization_data: pd.DataFrame) -> bool:
        """takes incoming_localizatoin_data, determines slot id, returns output of get_model_input
        for appropriate slot_df (the prediction)"""
        #assume that incoming_localization_data is a dataframe consisting of one row
        matching_slot_df = self.slot_dfs[incoming_localization_data.iloc[0]['fryer_slot_id']]
        pred_input_data = matching_slot_df.get_model_input(incoming_localization_data)
        return self.predictor.make_pred(pred_input_data)
    
    # def receive_new_row(self, new_row: pd.DataFrame) -> None:
    #     """takes new row, determines slot id, calls add_row method for appropriate
    #     slot_df"""
    #     #assume that new_row is a dataframe consisting of one row
    #     matching_slot_df = self.slot_dfs[new_row.iloc[0]['fryer_slot_id']]
    #     matching_slot_df.add_row(new_row)

