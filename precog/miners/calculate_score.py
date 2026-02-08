from typing import List, Tuple
import numpy as np
from pandas import DataFrame

import bittensor as bt
from precog import constants
from precog.protocol import Challenge
from precog.utils.cm_data import CMData
from precog.utils.general import pd_to_dict, rank
from precog.utils.timestamp import align_timepoints, get_before, mature_dictionary, to_datetime, to_str


def calc_rewards(
    self,
    synapse_timestamp : str,
    prediction_point: float,
    pre_interval : Tuple[float, float]
) -> float:
    evaluation_window_hours = constants.EVALUATION_WINDOW_HOURS
    prediction_future_hours = constants.PREDICTION_FUTURE_HOURS
    prediction_interval_minutes = constants.PREDICTION_INTERVAL_MINUTES

    expected_timepoints = evaluation_window_hours * 60 / prediction_interval_minutes
    bt.logging.debug(f"🛑 evaluation_window_hours: {evaluation_window_hours}  prediction_future_hours: {prediction_future_hours}  prediction_interval_minutes: {prediction_interval_minutes}")

    bt.logging.debug(f"🛑 synapse_timestamp: {synapse_timestamp}  prediction_point: {prediction_point}  pre_interval: {pre_interval[0]} {pre_interval[1]}")

    timestamp = synapse_timestamp
    cm = CMData()

    start_time: str =to_str(get_before(timestamp=timestamp, hours=evaluation_window_hours + prediction_future_hours))
    end_time: str = to_str(get_before(timestamp=timestamp, hours=prediction_future_hours))

    historical_price_data: DataFrame = cm.get_CM_ReferenceRate(
        assets="BTC", start=start_time, end=end_time, frequency="1s"
    )
    cm_data = pd_to_dict(historical_price_data)
    
    miner_history = MinerHistory(uid=0)  # Assuming uid is not used in this context
    miner_history.add_prediction(timestamp, prediction_point, pre_interval)
    # Get predictions from the evaluat`ion window that have had time to mature
    prediction_dict, interval_dict = miner_history.format_predictions(
        reference_timestamp = get_before(timestamp, hours = prediction_future_hours),
        hours = evaluation_window_hours,
    )
    bt.logging.debug(f"🛑 Prediction_dict:{prediction_dict} Interval_dict: {interval_dict}")
    # Mature the predictions (shift forward by 1 hour)
    mature_time_dict = mature_dictionary(prediction_dict, hours=prediction_future_hours)
    bt.logging.debug(f"🛑 Mature_time_dic:{mature_time_dict}")
    preds, price, aligned_pred_timestamps = align_timepoints(mature_time_dict, cm_data)
    bt.logging.debug(f"🛑 Preds :{preds}")
    num_predictions = len(preds) if preds is not None else 0

    completeness_ratio = min(num_predictions / expected_timepoints, 1.0)

    bt.logging.debug(f"🛑 completeness_ratio : {completeness_ratio}")

    inters, interval_prices, aligned_int_timestamps = align_timepoints(interval_dict, cm_data)
    bt.logging.debug(f"🛑 Inters: {inters}  Interval_prices : {interval_prices}")
    adjusted_interval_error= 0
    if any([np.isnan(inters).any(), np.isnan(interval_prices).any()]):
        adjusted_interval_error = 0
    else :
        base_interval_error = interval_error(inters, interval_prices)
        bt.logging.debug(f"🛑 Base_interval_error: {base_interval_error} ")
        adjusted_interval_error = base_interval_error * completeness_ratio
        bt.logging.debug(f"🛑 Adjusted_interval_error: {adjusted_interval_error} ")

    return adjusted_interval_error

def interval_error(intervals, cm_prices) -> float: 
    bt.logging.debug(f"🛑 Intervals: {intervals}  Interval-1: {intervals[:-1]} ")
    if intervals is None:
        return 0.0
    else: 
        f_w=0
        f_i=0
        interval_errors = []
        for i, interval_to_evaluate in enumerate(intervals[:-1]):
            lower_bound_prediction = np.min(interval_to_evaluate)
            upper_bound_prediction = np.max(interval_to_evaluate)
            effective_min = np.max([lower_bound_prediction, np.min(cm_prices[i+1 :])])
            effective_max = np.min([upper_bound_prediction, np.max(cm_prices[i+1 :])])
            if abs(upper_bound_prediction - lower_bound_prediction) <1e-10:
                f_w=0
            else: 
                f_w =max(0,(effective_max - effective_min) / (upper_bound_prediction - lower_bound_prediction)) 
            f_i = sum(
                (cm_prices[i+1:] >= lower_bound_prediction) & (cm_prices[i+1:] <= upper_bound_prediction)
            ) / len(cm_prices[i+1:])
            
            interval_errors.append(f_w * f_i)
        if not interval_errors:
            mean_error = 0
        elif len(interval_errors) == 1:
            mean_error = interval_errors[0]
        else : 
            mean_error = np.nanmean(np.array(interval_errors)).item()
        bt.logging.debug(f"🛑 f_w: {f_w:.3f}  f_i: {f_i:.3f}   mean_error: {mean_error:.3f}")
        return mean_error