from utils import read_yaml_to_dict, get_n_random_input_rows_from_csv
from orchestrate import MainDataFrame


CONFIG_DICT = read_yaml_to_dict("config.yaml")

CONFIG_DICT['SITE_ID'] = "wc26"
CONFIG_DICT['MISGRABS_OR_COLLISIONS'] = "collisions"
CONFIG_DICT["LOAD_CSV"] = True
NUM_TO_SAMPLE = 10



if CONFIG_DICT["LOAD_CSV"]:
    input_row_lst = get_n_random_input_rows_from_csv(10, CONFIG_DICT["CSV_PATH"])

main_data_frame = MainDataFrame(CONFIG_DICT)


for input_row in input_row_lst:
    print(main_data_frame.receive_incoming_localization_data(input_row))