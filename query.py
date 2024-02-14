
#!/usr/bin/env python

from sqlalchemy import create_engine, text
import pandas as pd
import time
from functools import reduce
import numpy as np
from utils import read_yaml_to_dict
from typing import Tuple

AUTH_DICT = read_yaml_to_dict("./mysql_auth.yaml")
sc_db_string = AUTH_DICT["sc_db_string"]
cl_db_string = AUTH_DICT["cl_db_string"]


def make_behaviors_table_df(db_string: str) -> pd.DataFrame:
    return pd.read_sql_table("behaviors", con=create_engine(db_string))

def coalesce(*arg):
  return reduce(lambda x, y: x if x is not None else y, arg)

BEHAVIORS_QUERY = """
with dataset as (
    select
        *,
        FROM_UNIXTIME(start_time) AS start_time_ts,
        FROM_UNIXTIME(end_time) AS end_time_ts,
        REPLACE(REPLACE(REPLACE(request, '''', '"'), 'True', 'true'), 'False', 'false') AS request_json,
        REPLACE(REPLACE(REPLACE(response, '''', '"'), 'True', 'true'), 'False', 'false') AS response_json
    from behaviors
    WHERE
        FROM_UNIXTIME(start_time) BETWEEN {start_time} AND {end_time}
        and request is not null
        and response is not null
        and request like '{{%'
        and response like '{{%'
),
bc_plan as (
    select
        behavior_name,
        req_uuid,
        fryer_slot_id,
        basket_id,
        CASE
            WHEN event_type = 'plan' and behavior_name like 'fryer_to_%' THEN 'frying'
            WHEN event_type = 'plan' and behavior_name like 'hanger_to_%' THEN 'hanging'
            ELSE ''
        END AS basket_state,
        CAST(json_extract(request_json, '$.basket_source.position.x') AS double) as "basket_source_x",
        CAST(json_extract(request_json, '$.basket_source.position.y') AS double) as "basket_source_y",
        CAST(json_extract(request_json, '$.basket_source.position.z') AS double) as "basket_source_z",
        CAST(json_extract(request_json, '$.basket_source.orientation.w') AS double) as "basket_source_qw",
        CAST(json_extract(request_json, '$.basket_source.orientation.x') AS double) as "basket_source_qx",
        CAST(json_extract(request_json, '$.basket_source.orientation.y') AS double) as "basket_source_qy",
        CAST(json_extract(request_json, '$.basket_source.orientation.z') AS double) as "basket_source_qz",
        CAST(json_extract(request_json, '$.fryer_slot.position.x') AS double) as "fryer_slot_x",
        CAST(json_extract(request_json, '$.fryer_slot.position.y') AS double) as "fryer_slot_y",
        CAST(json_extract(request_json, '$.fryer_slot.position.z') AS double) as "fryer_slot_z",
        CAST(json_extract(request_json, '$.fryer_slot.orientation.w') AS double) as "fryer_slot_qw",
        CAST(json_extract(request_json, '$.fryer_slot.orientation.x') AS double) as "fryer_slot_qx",
        CAST(json_extract(request_json, '$.fryer_slot.orientation.y') AS double) as "fryer_slot_qy",
        CAST(json_extract(request_json, '$.fryer_slot.orientation.z') AS double) as "fryer_slot_qz",
        CAST(json_extract(request_json, '$.hanger_tool_slot.position.x') AS double) as "hanger_tool_slot_x",
        CAST(json_extract(request_json, '$.hanger_tool_slot.position.y') AS double) as "hanger_tool_slot_y",
        CAST(json_extract(request_json, '$.hanger_tool_slot.position.z') AS double) as "hanger_tool_slot_z",
        CAST(json_extract(request_json, '$.hanger_tool_slot.orientation.w') AS double) as "hanger_tool_slot_qw",
        CAST(json_extract(request_json, '$.hanger_tool_slot.orientation.x') AS double) as "hanger_tool_slot_qx",
        CAST(json_extract(request_json, '$.hanger_tool_slot.orientation.y') AS double) as "hanger_tool_slot_qy",
        CAST(json_extract(request_json, '$.hanger_tool_slot.orientation.z') AS double) as "hanger_tool_slot_qz"
    from dataset
    where event_type = 'plan'
),
bc_exec_with_error_code as (
    select
        start_time_ts as exec_start_time_ts,
        end_time_ts as exec_end_time_ts,
        request_json as exec_request_json,
        response_json as exec_response_json,
        CAST(json_extract(response_json, '$.error.code') AS unsigned) AS exec_error_code,
        success as exec_success,
        req_uuid
    from dataset
    where event_type = 'exec'
),
bc_exec as (
    select
        *,
        CASE
            when not exec_success and exec_error_code = 103 then 'Misgrab'
            when not exec_success and exec_error_code = 100 then 'Collision'
            when not exec_success and exec_error_code = 4001 then 'FAILED_TO_ACQUIRE_BEHAVIOR_LOCK'
            when not exec_success and exec_error_code = 4008 then 'FAILED_TO_PLAN_BEHAVIOR'
            when not exec_success and exec_error_code = 4010 then 'SOURCE_TOO_FAR_FROM_STARTING_SLOT'
            when not exec_success and exec_error_code = 10002 then 'OBJECT_NOT_FOUND'
            ELSE 'no_failure'
        END AS failure_description
    from bc_exec_with_error_code
),
bca as (
    SELECT
        behavior_name,
        fryer_slot_id,
        basket_id,
        basket_state,
        basket_source_x,
        basket_source_y,
        basket_source_z,
        ATAN2(2.0*(basket_source_qw*basket_source_qx + basket_source_qy*basket_source_qz), 1 - (basket_source_qx*basket_source_qx + basket_source_qy*basket_source_qy)) as basket_source_roll,
        ASIN(2.0*(basket_source_qy*basket_source_qw - basket_source_qz*basket_source_qy)) as basket_source_pitch,
        ATAN2(2.0*(basket_source_qz*basket_source_qw + basket_source_qx*basket_source_qy), -1.0 + 2.0*(basket_source_qw*basket_source_qw + basket_source_qx*basket_source_qx)) as basket_source_yaw,
        fryer_slot_x,
        fryer_slot_y,
        fryer_slot_z,
        ATAN2(2.0*(fryer_slot_qw*fryer_slot_qx + fryer_slot_qy*fryer_slot_qz), 1 - (fryer_slot_qx*fryer_slot_qx + fryer_slot_qy*fryer_slot_qy)) as fryer_slot_roll,
        ASIN(2.0*(fryer_slot_qy*fryer_slot_qw - fryer_slot_qz*fryer_slot_qy)) as fryer_slot_pitch,
        ATAN2(2.0*(fryer_slot_qz*fryer_slot_qw + fryer_slot_qx*fryer_slot_qy), -1.0 + 2.0*(fryer_slot_qw*fryer_slot_qw + fryer_slot_qx*fryer_slot_qx)) as fryer_slot_yaw,
        hanger_tool_slot_x,
        hanger_tool_slot_y,
        hanger_tool_slot_z,
        ATAN2(2.0*(hanger_tool_slot_qw*hanger_tool_slot_qx + hanger_tool_slot_qy*hanger_tool_slot_qz), 1 - (hanger_tool_slot_qx*hanger_tool_slot_qx + hanger_tool_slot_qy*hanger_tool_slot_qy)) as hanger_tool_slot_roll,
        ASIN(2.0*(hanger_tool_slot_qy*hanger_tool_slot_qw - hanger_tool_slot_qz*hanger_tool_slot_qy)) as hanger_tool_slot_pitch,
        ATAN2(2.0*(hanger_tool_slot_qz*hanger_tool_slot_qw + hanger_tool_slot_qx*hanger_tool_slot_qy), -1.0 + 2.0*(hanger_tool_slot_qw*hanger_tool_slot_qw + hanger_tool_slot_qx*hanger_tool_slot_qx)) as hanger_tool_slot_yaw,
        #
        exec_start_time_ts as behavior_start_time,
        exec_error_code as error_code,
        exec_success,
        failure_description
    FROM bc_plan
    JOIN bc_exec
    USING (req_uuid)
)
select * from bca
limit 1000
"""

OFFSETS_QUERY = """
with offsets_base as (
    SELECT
        offset_atlas_time,
        offset_path,
        cast(basket_id as unsigned) as basket_id,
        point_x,
        point_y,
        point_z,
        JSON_OBJECT(
            'x', CAST(point_x AS double),
            'y', CAST(point_y AS double),
            'z', CAST(point_z AS double)
        ) AS "offset",
        reporter,
        CAST(SUBSTRING(SUBSTRING_INDEX(SUBSTRING_INDEX(offset_path, '/', 2), '/', -1), 7) as unsigned) AS fryer,
        SUBSTRING_INDEX(SUBSTRING_INDEX(offset_path, '/', 3), '/', -1) AS side,
        SUBSTRING_INDEX(SUBSTRING_INDEX(offset_path, '/', 4), '/', -1) AS slot,
        SUBSTRING_INDEX(SUBSTRING_INDEX(offset_path, '/', 5), '/', -1) AS reference_type,
        waypoint
    FROM
        offset_atlas
    WHERE
        offset_atlas_time BETWEEN {start_time} AND {end_time}
        and (CHAR_LENGTH(offset_path) - CHAR_LENGTH(REPLACE(offset_path, '/', '')) + 1) >= 5

),
offsets as (
    select
        offset_atlas_time,
        offset_path,
        basket_id,
        offset,
        reporter,
        fryer,
        side,
        slot,
        reference_type,
        waypoint,
        point_x,
        point_y,
        point_z,
        CASE
            WHEN side = 'left_slot' THEN (fryer - 1) * 2 + 1
            WHEN side = 'right_slot' THEN (fryer - 1) * 2 + 2
        END AS fryer_slot_id,
        CASE
            WHEN slot LIKE '%frying_slot' THEN 'frying'
            WHEN slot LIKE '%hanging_slot' THEN 'hanging'
            ELSE ''
        END as basket_state
    from offsets_base
)
select * from offsets
"""





def query_behaviors(start_time: str, end_time: str) -> pd.DataFrame:
    simple_chef_db_con = create_engine(sc_db_string)
    behaviors_query = BEHAVIORS_QUERY.format(start_time=start_time, end_time=end_time)
    behaviors_df = pd.read_sql_query(text(behaviors_query), con=simple_chef_db_con)
    return behaviors_df

def query_offsets(start_time: str, end_time: str) -> pd.DataFrame:
    chippy_log_db_con = create_engine(cl_db_string)
    offsets_query = OFFSETS_QUERY.format(start_time=start_time, end_time=end_time)
    offsets_df = pd.read_sql_query(text(offsets_query), con=chippy_log_db_con)
    return offsets_df

def get_model_rows(behaviors_start_time: str, behaviors_end_time: str, offset_start_time: str, offset_end_time: str) -> Tuple[pd.DataFrame]:
    behaviors_df = query_behaviors(behaviors_start_time, behaviors_end_time)
    offsets_df = query_offsets(offset_start_time, offset_end_time)
    return behaviors_df, offsets_df



def combine_both_dfs(behaviors_df: pd.DataFrame, offsets_df: pd.DataFrame, behaviors_start_time: str, behaviors_end_time: str, offsets_start_time: str, offsets_end_time: str) -> pd.DataFrame:
    offsets_df.drop_duplicates(keep='first', inplace=True)

    offsets_df['basket_id'] = offsets_df['basket_id'].astype(float)
    offsets_df['fryer_slot_id'] = offsets_df['fryer_slot_id'].astype(int)

    if behaviors_df.empty:
        print("behaviors_df is empty")
        exit()

    extra_columns = [
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
    for col in extra_columns:
        behaviors_df[col] = np.nan

    for i, b_row in behaviors_df.iterrows():
        # Make df that has matches on fryer_slot_id and filter by time
        offsets_filtered = offsets_df[
            (offsets_df['fryer_slot_id'] == b_row['fryer_slot_id'])
            & (offsets_df['offset_atlas_time'] < b_row['behavior_start_time'])
        ]

        # offsets_filtered_basket_51 = offsets_filtered[offsets_filtered['basket_id'] == 51]
        offsets_filtered = offsets_filtered.sort_values(by='offset_atlas_time', ascending=False)

        target = offsets_filtered[offsets_filtered['waypoint'] == 'target']

        # Make fryer_target absolute and relative dfs
        # Match on reference_type for fryer_target, absolute and relative
        w_state = target[target['basket_state'] == 'frying']
        ft_abs = w_state[w_state['reference_type'] == 'absolute']
        ft_rel = w_state[w_state['reference_type'] == 'relative']
        # Try matching on basket_id. if empty, use default offsets
        ft_abs_bid = ft_abs[ft_abs['basket_id'] == b_row['basket_id']]
        ft_rel_bid = ft_rel[ft_rel['basket_id'] == b_row['basket_id']]
        if ft_abs_bid.empty:
            ft_abs = ft_abs[ft_abs['basket_id'].isnull()]
        else:
            ft_abs = ft_abs_bid
        if ft_rel_bid.empty:
            ft_rel = ft_rel[ft_rel['basket_id'].isnull()]
        else:
            ft_rel = ft_rel_bid

        # Make hanger_target absolute and relative dfs
        # Match on reference_type for hanger_target, absolute and relative
        w_state = target[target['basket_state'] == 'hanging']
        ht_abs = w_state[w_state['reference_type'] == 'absolute']
        ht_rel = w_state[w_state['reference_type'] == 'relative']
        # Try matching on basket_id. if empty, use default offsets
        ht_abs_bid = ht_abs[ht_abs['basket_id'] == b_row['basket_id']]
        ht_rel_bid = ht_rel[ht_rel['basket_id'] == b_row['basket_id']]
        if ht_abs_bid.empty:
            ht_abs = ht_abs[ht_abs['basket_id'].isnull()]
        else:
            ht_abs = ht_abs_bid
        if ht_rel_bid.empty:
            ht_rel = ht_rel[ht_rel['basket_id'].isnull()]
        else:
            ht_rel = ht_rel_bid

        # Make basket_source absolute and relative dfs
        # Match on reference_type for basket_source, absolute and relative
        bs_offsets = offsets_filtered[
            (offsets_filtered['waypoint'] == 'source') &
            (offsets_filtered['basket_state'] == b_row['basket_state'])
        ]
        bs_abs = bs_offsets[bs_offsets['reference_type'] == 'absolute']
        bs_rel = bs_offsets[bs_offsets['reference_type'] == 'relative']

        # Try matching on basket_id. if empty, use default offsets
        bs_abs_bid = bs_abs[bs_abs['basket_id'] == b_row['basket_id']]
        bs_rel_bid = bs_rel[bs_rel['basket_id'] == b_row['basket_id']]
        if bs_abs_bid.empty:
            bs_abs = bs_abs[bs_abs['basket_id'].isnull()]
        else:
            bs_abs = bs_abs_bid
        if bs_rel_bid.empty:
            bs_rel = bs_rel[bs_rel['basket_id'].isnull()]
        else:
            bs_rel = bs_rel_bid

        # If missing_offset, skip row
        missing_offset = False
        if ft_abs.empty:
            print(f"no ft_abs offset for {b_row['fryer_slot_id']}, {b_row['basket_id']}")
            missing_offset |= True
        if ft_rel.empty:
            print(f"no ft_rel offset for {b_row['fryer_slot_id']}, {b_row['basket_id']}")
            missing_offset |= True
        if ht_abs.empty:
            print(f"no ht_abs offset for {b_row['fryer_slot_id']}, {b_row['basket_id']}")
            missing_offset |= True
        if ht_rel.empty:
            print(f"no ht_rel offset for {b_row['fryer_slot_id']}, {b_row['basket_id']}")
            missing_offset |= True
        if bs_abs.empty:
            print(f"no bs_abs offset for {b_row['fryer_slot_id']}, {b_row['basket_id']}")
            missing_offset |= True
        if bs_rel.empty:
            print(f"no bs_rel offset for {b_row['fryer_slot_id']}, {b_row['basket_id']}")
            missing_offset |= True
        if missing_offset:
            continue

        # Sort by offset_atlas_time and take the first row
        ft_abs = ft_abs.sort_values(by='offset_atlas_time', ascending=False).iloc[0]
        ft_rel = ft_rel.sort_values(by='offset_atlas_time', ascending=False).iloc[0]
        ht_abs = ht_abs.sort_values(by='offset_atlas_time', ascending=False).iloc[0]
        ht_rel = ht_rel.sort_values(by='offset_atlas_time', ascending=False).iloc[0]
        bs_abs = bs_abs.sort_values(by='offset_atlas_time', ascending=False).iloc[0]
        bs_rel = bs_rel.sort_values(by='offset_atlas_time', ascending=False).iloc[0]

        b_row['fryer_slot_x_offset_abs'] = coalesce(ft_abs['point_x'], 0.0)
        b_row['fryer_slot_x_offset_rel'] = coalesce(ft_rel['point_x'], 0.0)
        b_row['fryer_slot_y_offset_abs'] = coalesce(ft_abs['point_y'], 0.0)
        b_row['fryer_slot_y_offset_rel'] = coalesce(ft_rel['point_y'], 0.0)
        b_row['fryer_slot_z_offset_abs'] = coalesce(ft_abs['point_z'], 0.0)
        b_row['fryer_slot_z_offset_rel'] = coalesce(ft_rel['point_z'], 0.0)

        b_row['hanger_tool_slot_x_offset_abs'] = coalesce(ht_abs['point_x'], 0.0)
        b_row['hanger_tool_slot_x_offset_rel'] = coalesce(ht_rel['point_x'], 0.0)
        b_row['hanger_tool_slot_y_offset_abs'] = coalesce(ht_abs['point_y'], 0.0)
        b_row['hanger_tool_slot_y_offset_rel'] = coalesce(ht_rel['point_y'], 0.0)
        b_row['hanger_tool_slot_z_offset_abs'] = coalesce(ht_abs['point_z'], 0.0)
        b_row['hanger_tool_slot_z_offset_rel'] = coalesce(ht_rel['point_z'], 0.0)

        b_row['basket_source_x_offset_abs'] = coalesce(bs_abs['point_x'], 0.0)
        b_row['basket_source_x_offset_rel'] = coalesce(bs_rel['point_x'], 0.0)
        b_row['basket_source_y_offset_abs'] = coalesce(bs_abs['point_y'], 0.0)
        b_row['basket_source_y_offset_rel'] = coalesce(bs_rel['point_y'], 0.0)
        b_row['basket_source_z_offset_abs'] = coalesce(bs_abs['point_z'], 0.0)
        b_row['basket_source_z_offset_rel'] = coalesce(bs_rel['point_z'], 0.0)

        # Apply offsets to row
        # TODO: Correct how relative offsets are applied. Currently they are applied same as absolute.
        # They should be applied relative to orientation of frame they're applied to.
        b_row['fryer_slot_x_no_offset'] = b_row['fryer_slot_x'] - b_row['fryer_slot_x_offset_abs'] - b_row['fryer_slot_x_offset_rel']
        b_row['fryer_slot_y_no_offset'] = b_row['fryer_slot_y'] - b_row['fryer_slot_y_offset_abs'] - b_row['fryer_slot_y_offset_rel']
        b_row['fryer_slot_z_no_offset'] = b_row['fryer_slot_z'] - b_row['fryer_slot_z_offset_abs'] - b_row['fryer_slot_z_offset_rel']
        b_row['fryer_slot_roll_no_offset'] = b_row['fryer_slot_roll']
        b_row['fryer_slot_pitch_no_offset'] = b_row['fryer_slot_pitch']
        b_row['fryer_slot_yaw_no_offset'] = b_row['fryer_slot_yaw']
        b_row['hanger_tool_slot_x_no_offset'] = b_row['hanger_tool_slot_x'] - b_row['hanger_tool_slot_x_offset_abs'] - b_row['hanger_tool_slot_x_offset_rel']
        b_row['hanger_tool_slot_y_no_offset'] = b_row['hanger_tool_slot_y'] - b_row['hanger_tool_slot_y_offset_abs'] - b_row['hanger_tool_slot_y_offset_rel']
        b_row['hanger_tool_slot_z_no_offset'] = b_row['hanger_tool_slot_z'] - b_row['hanger_tool_slot_z_offset_abs'] - b_row['hanger_tool_slot_z_offset_rel']
        b_row['hanger_tool_slot_roll_no_offset'] = b_row['hanger_tool_slot_roll']
        b_row['hanger_tool_slot_pitch_no_offset'] = b_row['hanger_tool_slot_pitch']
        b_row['hanger_tool_slot_yaw_no_offset'] = b_row['hanger_tool_slot_yaw']
        b_row['basket_source_x_no_offset'] = b_row['basket_source_x'] - b_row['basket_source_x_offset_abs'] - b_row['basket_source_x_offset_rel']
        b_row['basket_source_y_no_offset'] = b_row['basket_source_y'] - b_row['basket_source_y_offset_abs'] - b_row['basket_source_y_offset_rel']
        b_row['basket_source_z_no_offset'] = b_row['basket_source_z'] - b_row['basket_source_z_offset_abs'] - b_row['basket_source_z_offset_rel']
        b_row['basket_source_roll_no_offset'] = b_row['basket_source_roll']
        b_row['basket_source_pitch_no_offset'] = b_row['basket_source_pitch']
        b_row['basket_source_yaw_no_offset'] = b_row['basket_source_yaw']

        behaviors_df.loc[i, b_row.index] = b_row

    # behaviors_df.to_csv("behaviors_df2.csv")
    return behaviors_df

def get_combined_data(behaviors_start_time: str, behaviors_end_time: str, offsets_start_time: str, offsets_end_time: str) -> pd.DataFrame:
    start = time.time()
    behaviors_df, offsets_df = get_model_rows(behaviors_start_time, behaviors_end_time, offsets_start_time, offsets_end_time)
    combined_df = combine_both_dfs(behaviors_df, offsets_df, behaviors_start_time, behaviors_end_time, offsets_start_time, offsets_end_time)
    end = time.time()
    print(end - start)
    return combined_df

def convert_str_time(str_time: str) -> str:
    new_str = f"STR_TO_DATE('{str_time}', '%Y-%m-%d %h:%m:%s')"
    print(new_str)
    return new_str

offsets_start_time = "CURRENT_TIMESTAMP - INTERVAL '20' DAY"
offsets_end_time = "CURRENT_TIMESTAMP - INTERVAL '3' DAY"
behaviors_start_time = "CURRENT_TIMESTAMP - INTERVAL '10' DAY"
behaviors_end_time = "CURRENT_TIMESTAMP - INTERVAL '3' DAY"

offsets_start_time = convert_str_time('2024-01-05')#"STR_TO_DATE('2024-02-17', '%Y-%m-%d %h:%m:%s')"#"CURRENT_TIMESTAMP - INTERVAL '20' DAY"
offsets_end_time = "CURRENT_TIMESTAMP - INTERVAL '3' DAY"
# behaviors_start_time = "CURRENT_TIMESTAMP - INTERVAL '7' DAY"
behaviors_start_time = convert_str_time('2024-02-06')#"STR_TO_DATE('2024-02-17', '%Y-%m-%d %h:%m:%s')"#"UNIX_TIMESTAMP(STR_TO_DATE('2024-02-05', '%Y-%m-%d'))"
behaviors_end_time = "CURRENT_TIMESTAMP - INTERVAL '3' DAY"

ab = get_combined_data(behaviors_start_time, behaviors_end_time, offsets_start_time, offsets_end_time)
print(len(ab))