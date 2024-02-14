from utils import read_yaml_to_dict, get_n_random_input_rows_from_csv, convert_str_time
from orchestrate import MainDataFrame


CONFIG_DICT = read_yaml_to_dict("config.yaml")

CONFIG_DICT['SITE_ID'] = "wc26"
CONFIG_DICT['MISGRABS_OR_COLLISIONS'] = "collisions"
CONFIG_DICT["LOAD_CSV"] = False

#start and end time for main query
# CONFIG_DICT["INIT_START_TIME"] = "CURRENT_TIMESTAMP - INTERVAL '20' DAY"#"2023-01-01 08:00:00"
# CONFIG_DICT["INIT_END_TIME"] = "CURRENT_TIMESTAMP - INTERVAL '3' DAY"#"2024-02-12 08:00:00"

# CONFIG_DICT["BEHAVIORS_INIT_START_TIME"] = "CURRENT_TIMESTAMP - INTERVAL '20' DAY"
# CONFIG_DICT["BEHAVIORS_INIT_END_TIME"] = "CURRENT_TIMESTAMP - INTERVAL '3' DAY"
# CONFIG_DICT["OFFSETS_INIT_START_TIME"] = "CURRENT_TIMESTAMP - INTERVAL '25' DAY"
# CONFIG_DICT["OFFSETS_INIT_END_TIME"] = "CURRENT_TIMESTAMP - INTERVAL '3' DAY"

CONFIG_DICT["BEHAVIORS_INIT_START_TIME"] = "CURRENT_TIMESTAMP - INTERVAL '500' DAY"
CONFIG_DICT["BEHAVIORS_INIT_END_TIME"] = "CURRENT_TIMESTAMP - INTERVAL '30' DAY"
CONFIG_DICT["OFFSETS_INIT_START_TIME"] = "CURRENT_TIMESTAMP - INTERVAL '550' DAY"
CONFIG_DICT["OFFSETS_INIT_END_TIME"] = "CURRENT_TIMESTAMP - INTERVAL '30' DAY"


# CONFIG_DICT["BEHAVIORS_INIT_START_TIME"] = convert_str_time("2024-01-01-08:00:00")
# CONFIG_DICT["BEHAVIORS_INIT_END_TIME"] = convert_str_time("2024-02-01-08:00:00")
# CONFIG_DICT["OFFSETS_INIT_START_TIME"] = convert_str_time("2024-01-01-08:00:00")
# CONFIG_DICT["OFFSETS_INIT_END_TIME"] = convert_str_time("2024-02-01-08:00:00")


NUM_TO_SAMPLE = 10




# if CONFIG_DICT["LOAD_CSV"]:
# input_row_lst = get_n_random_input_rows_from_csv(10, CONFIG_DICT["CSV_PATH"])
main_data_frame = MainDataFrame(CONFIG_DICT)

input_row_lst = main_data_frame.get_n_random_rows(NUM_TO_SAMPLE)


for input_row in input_row_lst:
    print(main_data_frame.receive_incoming_localization_data(input_row))