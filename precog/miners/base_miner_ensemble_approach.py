import time
from typing import Tuple

import bittensor as bt
import pandas as pd

from precog.protocol import Challenge
from precog.utils.cm_data import CMData
from precog.utils.timestamp import get_before, to_datetime, to_str


def get_point_estimate(cm: CMData, timestamp: str) -> float:
    """Make a naive forecast by predicting the most recent price

    Args:
        cm (CMData): The CoinMetrics API client
        timestamp (str): The current timestamp provided by the validator request

    Returns:
        (float): The current BTC price tied to the provided timestamp
    """

    # Ensure timestamp is correctly typed and set to UTC
    provided_timestamp = to_datetime(timestamp)

    # Query CM API for a pandas dataframe with only one record
    price_data: pd.DataFrame = cm.get_CM_ReferenceRate(
        assets="BTC",
        start=None,
        end=to_str(provided_timestamp),
        frequency="1s",
        limit_per_asset=1,
        paging_from="end",
        use_cache=False,
    )

    # Get current price closest to the provided timestamp
    btc_price: float = float(price_data["ReferenceRateUSD"].iloc[-1])

    # Return the current price of BTC as our point estimate
    return btc_price

# Naive prediction Interval
def naive_prediction_interval(cm: CMData, timestamp: str, point_estimate: float) -> Tuple[float, float]:
    """Make a naive multi-step prediction interval by estimating
    the sample standard deviation

    Args:
        cm (CMData): The CoinMetrics API client
        timestamp (str): The current timestamp provided by the validator request
        point_estimate (float): The center of the prediction interval

    Returns:
        (float): The 90% naive prediction interval lower bound
        (float): The 90% naive prediction interval upper bound

    Notes:
        Make reasonable assumptions that the 1s BTC price residuals are
        uncorrelated and normally distributed
    """

    # Set the time range to be 24 hours
    # Ensure both timestamps are correctly typed and set to UTC
    start_time = get_before(timestamp, days=1, minutes=0, seconds=0)
    end_time = to_datetime(timestamp)

    # Query CM API for sample standard deviation of the 1s residuals
    historical_price_data: pd.DataFrame = cm.get_CM_ReferenceRate(
        assets="BTC", start=to_str(start_time), end=to_str(end_time), frequency="1s"
    )
    residuals: pd.Series = historical_price_data["ReferenceRateUSD"].diff()
    sample_std_dev: float = float(residuals.std())

    # We have the standard deviation of the 1s residuals
    # We are forecasting forward 60m, which is 3600s
    # We must scale the 1s sample standard deviation to reflect a 3600s forecast
    # Make reasonable assumptions that the 1s residuals are uncorrelated and normally distributed
    # To do this naively, we multiply the std dev by the square root of the number of time steps
    time_steps: int = 3600
    naive_forecast_std_dev: float = sample_std_dev * (time_steps**0.5)

    # For a 90% prediction interval, we use the coefficient 1.64
    # Make reasonable assumptions that the 1s residuals are uncorrelated and normally distributed
    coefficient: float = 1.64

    # Calculate the lower bound and upper bound
    lower_bound: float = point_estimate - coefficient * naive_forecast_std_dev
    upper_bound: float = point_estimate + coefficient * naive_forecast_std_dev

    # Return the naive prediction interval for our forecast
    return lower_bound, upper_bound


# GARCH Models for Volatility Clustering
def garch_prediction_interval(cm: CMData, timestamp: str, point_estimate: float) -> Tuple[float, float]:
    from arch import arch_model

    """Make a naive multi-step prediction interval by estimating
    the sample standard deviation

    Args:
        cm (CMData): The CoinMetrics API client
        timestamp (str): The current timestamp provided by the validator request
        point_estimate (float): The center of the prediction interval

    Returns:
        (float): The 90% naive prediction interval lower bound
        (float): The 90% naive prediction interval upper bound

    Notes:
        Make reasonable assumptions that the 1s BTC price residuals are
        uncorrelated and normally distributed
    """

    # Set the time range to be 24 hours
    # Ensure both timestamps are correctly typed and set to UTC
    start_time = get_before(timestamp, days=1, minutes=0, seconds=0)
    end_time = to_datetime(timestamp)

    # Query CM API for sample standard deviation of the 1s residuals
    historical_price_data: pd.DataFrame = cm.get_CM_ReferenceRate(
        assets="BTC", start=to_str(start_time), end=to_str(end_time), frequency="1s"
    )
    
    returns = 100 * historical_price_data["ReferenceRateUSD"].pct_change().dropna()

    # Fit GARCH model
    model = arch_model(returns, vol="GARCH", p=1, q=1)
    model_fit = model.fit(disp="off")

    # Forecase volatility for next 60 minutes
    forecast = model_fit.forecast(horizon = 60)
    forecast_variance = forecast.variance.iloc[-1].sum()
    forecast_std_dev = forecast_variance ** 0.5

    # Scale to price level
    price_level = historical_price_data["ReferenceRateUSD"].iloc[-1]
    price_std_dev = price_level * forecast_std_dev / 100

    coefficient = 1.645  # 90% confidence interval
    return point_estimate - coefficient * price_std_dev, point_estimate + coefficient * price_std_dev

# Empirical Prediction Interval
def empirical_prediction_interval(cm: CMData, timestamp: str, point_estimate: float) -> Tuple[float, float]:
    
    """Make a naive multi-step prediction interval by estimating
    the sample standard deviation

    Args:
        cm (CMData): The CoinMetrics API client
        timestamp (str): The current timestamp provided by the validator request
        point_estimate (float): The center of the prediction interval

    Returns:
        (float): The 90% naive prediction interval lower bound
        (float): The 90% naive prediction interval upper bound

    Notes:
        Make reasonable assumptions that the 1s BTC price residuals are
        uncorrelated and normally distributed
    """

    # Set the time range to be 24 hours
    # Ensure both timestamps are correctly typed and set to UTC
    start_time = get_before(timestamp, days=30, minutes=0, seconds=0)
    end_time = to_datetime(timestamp)

    # Query CM API for sample standard deviation of the 1s residuals
    historical_price_data: pd.DataFrame = cm.get_CM_ReferenceRate(
        assets="BTC", start=to_str(start_time), end=to_str(end_time), frequency="1h"
    )
     
    # Calculate hourly returns
    hourly_returns = historical_price_data["ReferenceRateUSD"].pct_change().dropna()

    # Calculate empirical percentiles for 60 minute returns

    lower_percentile = hourly_returns.quantile(0.05)
    upper_percentile = hourly_returns.quantile(0.95)

    # Calculate prediction interval
    lower_bound = point_estimate * (1 + lower_percentile)
    upper_bound = point_estimate * (1 + upper_percentile)

    return lower_bound, upper_bound

def get_prediction_interval(cm: CMData, timestamp: str, point_estimate: float) -> Tuple[float, float]:
    
    """Make a naive multi-step prediction interval by estimating
    the sample standard deviation

    Args:
        cm (CMData): The CoinMetrics API client
        timestamp (str): The current timestamp provided by the validator request
        point_estimate (float): The center of the prediction interval

    Returns:
        (float): The 90% naive prediction interval lower bound
        (float): The 90% naive prediction interval upper bound

    Notes:
        Make reasonable assumptions that the 1s BTC price residuals are
        uncorrelated and normally distributed
    """

    # Calculate intervals using different methods
    naive_lower, naive_upper = naive_prediction_interval(cm, timestamp, point_estimate)
    empirical_lower, empirical_upper = empirical_prediction_interval(cm, timestamp, point_estimate)

    # Try to use GARCH if possible, otherwise use a default
    try:
        garch_lower, garch_upper = garch_prediction_interval(cm, timestamp, point_estimate)
    except: 
        # If GARCH fails, use naive method as fallback
        garch_lower, garch_upper = naive_lower, naive_upper

    # Combine the predictions (simple average)
    combined_lower = (naive_lower + empirical_lower + garch_lower) / 3
    combined_upper = (naive_upper + empirical_upper + garch_upper) / 3

    return combined_lower, combined_upper



def forward(synapse: Challenge, cm: CMData) -> Challenge:
    total_start_time = time.perf_counter()
    bt.logging.info(
        f"👈 Received prediction request from: {synapse.dendrite.hotkey} for timestamp: {synapse.timestamp}"
    )

    point_estimate_start = time.perf_counter()
    # Get the naive point estimate
    point_estimate: float = get_point_estimate(cm=cm, timestamp=synapse.timestamp)

    point_estimate_time = time.perf_counter() - point_estimate_start
    bt.logging.debug(f"⏱️ Point estimate function took: {point_estimate_time:.3f} seconds")

    interval_start = time.perf_counter()
    # Get the naive prediction interval
    prediction_interval: Tuple[float, float] = get_prediction_interval(
        cm=cm, timestamp=synapse.timestamp, point_estimate=point_estimate
    )

    interval_time = time.perf_counter() - interval_start
    bt.logging.debug(f"⏱️ Prediction interval function took: {interval_time:.3f} seconds")

    synapse.prediction = point_estimate
    synapse.interval = prediction_interval

    total_time = time.perf_counter() - total_start_time
    bt.logging.debug(f"⏱️ Total forward call took: {total_time:.3f} seconds")

    if synapse.prediction is not None:
        bt.logging.success(f"Predicted price: {synapse.prediction}  |  Predicted Interval: {synapse.interval}")
    else:
        bt.logging.info("No prediction for this request.")
    return synapse
