import pandas as pd
import os
import datetime
import numpy as np
import multiprocessing as mp


def get_all_symbol_list(
    train_data_path: str
) -> list[str]:
    """
    Retrieves a list of currency code symbols from the training
    data directory.

    Returns:
        list: A list of currency code symbols extracted from the
        filenames in the training data directory.
    """
    parquet_name_list = os.listdir(train_data_path)
    symbol_list = [parquet_name.split(".")[0] for
                   parquet_name in parquet_name_list]
    return symbol_list


def get_single_symbol_kline_data(
    train_data_path: str,
    symbol: str
) -> pd.DataFrame:
    """
    Retrieves K-line (candlestick) data for a given cryptocurrency symbol
    from a Parquet file.

    Reads the Parquet file for the symbol, sets "timestamp" as index,
    converts all data to float64, and calculates VWAP. Infinite VWAP
    values are replaced with NaN and forward-filled. If an error occurs,
    returns an empty DataFrame.

    Args:
        symbol (str): Cryptocurrency symbol to retrieve K-line data.

    Returns:
        pd.DataFrame: Processed K-line data with VWAP, or empty DataFrame
        if an error occurs.
    """
    try:
        df = pd.read_parquet(f"{train_data_path}/{symbol}.parquet",
                             engine='fastparquet')
        # set the DataFrame's index to the "timestamp" column
        df = df.set_index("timestamp")
        # convert the data to 64-bit floating-point type.
        df = df.astype(np.float64)
        # calculate the volume-weighted average price (VWAP),handle
        # infinite values, and fill them with the previous valid value
        df['vwap'] = (df['amount'] / df['volume']).replace(
            [np.inf, -np.inf], np.nan).ffill()
        # calculate return
        df['return'] = df['close_price'].pct_change()
    except Exception as e:
        print(f"get_single_symbol_kline_data error: {e}")
        df = pd.DataFrame()
    return df


def get_all_symbol_kline(
    train_data_path: str
) -> tuple[
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray
]:
    """
    Retrieves and aggregates K-line (candlestick) data for all symbols
    in the training data directory using parallel processing.

    For each symbol, reads its K-line data and extracts the following
    columns: open_price, high_price, low_price, close_price, vwap, and
    amount. The results are concatenated into arrays for each field,
    sorted by timestamp.

    Returns:
        tuple: (
            all_symbol_list (list[str]): List of all symbol names,
            time_arr (np.ndarray): Array of timestamps as datetime objects,
            open_price_arr (np.ndarray): Array of open prices,
            high_price_arr (np.ndarray): Array of high prices,
            low_price_arr (np.ndarray): Array of low prices,
            close_price_arr (np.ndarray): Array of close prices,
            vwap_arr (np.ndarray): Array of VWAP values,
            amount_arr (np.ndarray): Array of trading amounts
        )
    """

    t0 = datetime.datetime.now()
    # Create a process pool for parallel processing
    pool = mp.Pool(mp.cpu_count() - 2)
    all_symbol_list = get_all_symbol_list(train_data_path)

    # Launch asynchronous tasks to read K-line data for each symbol
    df_list = [
        pool.apply_async(get_single_symbol_kline_data, (train_data_path, symbol))
        for symbol in all_symbol_list
    ]

    pool.close()
    pool.join()

    # Aggregate each price/amount column across all symbols
    df_open_price = pd.concat(
        [res.get()['open_price'] for res in df_list], axis=1
    ).sort_index(ascending=True)
    time_arr = pd.to_datetime(
        pd.Series(df_open_price.index), unit="ms"
    ).values
    open_price_arr = df_open_price.values.astype(float)
    high_price_arr = pd.concat(
        [res.get()['high_price'] for res in df_list], axis=1
    ).sort_index(ascending=True).values
    low_price_arr = pd.concat(
        [res.get()['low_price'] for res in df_list], axis=1
    ).sort_index(ascending=True).values
    close_price_arr = pd.concat(
        [res.get()['close_price'] for res in df_list], axis=1
    ).sort_index(ascending=True).values
    vwap_arr = pd.concat(
        [res.get()['vwap'] for res in df_list], axis=1
    ).sort_index(ascending=True).values
    return_arr = pd.concat(
        [res.get()['return'] for res in df_list], axis=1
    ).sort_index(ascending=True).values
    amount_arr = pd.concat(
        [res.get()['amount'] for res in df_list], axis=1
    ).sort_index(ascending=True).values

    print(
        f"Finished get_all_symbol_kline, "
        f"time elapsed: {datetime.datetime.now() - t0}"
    )
    return (
        all_symbol_list, time_arr, open_price_arr, high_price_arr,
        low_price_arr, close_price_arr, vwap_arr, amount_arr,
        return_arr
    )


def weighted_spearmanr(y_true, y_pred):
    """
    Calculate the weighted Spearman correlation coefficient.

    Steps:
    1. Rank y_true and y_pred in descending order (rank=1 is max value).
    2. Normalize ranks to [-1, 1], then square for weights.
    3. Compute weighted Pearson correlation on ranks.
    """
    n = len(y_true)
    r_true = pd.Series(y_true).rank(ascending=False, method='average')
    r_pred = pd.Series(y_pred).rank(ascending=False, method='average')

    x = 2 * (r_true - 1) / (n - 1) - 1
    w = x ** 2

    w_sum = w.sum()
    mu_true = (w * r_true).sum() / w_sum
    mu_pred = (w * r_pred).sum() / w_sum

    cov = (w * (r_true - mu_true) * (r_pred - mu_pred)).sum()
    var_true = (w * (r_true - mu_true) ** 2).sum()
    var_pred = (w * (r_pred - mu_pred) ** 2).sum()

    return cov / np.sqrt(var_true * var_pred)


def calculate_return_correlation_matrix(
    train_data_path='kline_data/train_data',
    output_csv='/Users/vasudev/Desktop/Projects/avenir-hku-web/return_correlation_matrix.csv'
) -> pd.DataFrame:
    """
    Load all crypto data, calculate return correlation matrix, and save to CSV.

    Args:
        train_data_path (str): Path to training data directory
        output_csv (str): Output CSV filename for correlation matrix

    Returns:
        pd.DataFrame: Correlation matrix of return values
    """
    print("Loading all cryptocurrency data...")

    # Get all symbols and load their data
    (
        all_symbol_list,
        time_arr,
        open_price_arr,
        high_price_arr,
        low_price_arr,
        close_price_arr,
        vwap_arr,
        amount_arr,
        return_arr,
    ) = get_all_symbol_kline(train_data_path)

    print(f"Loaded data for {len(all_symbol_list)} symbols")
    print(f"Data shape: {return_arr.shape}")

    # Create DataFrame with return data
    return_df = pd.DataFrame(return_arr, columns=all_symbol_list,
                             index=pd.to_datetime(time_arr))

    print("Calculating correlation matrix...")
    # Calculate correlation matrix
    correlation_matrix = return_df.corr()

    print(correlation_matrix)

    # Save to CSV
    correlation_matrix.to_csv(output_csv)
    print(f"Correlation matrix saved to {output_csv}")
    print(f"Matrix dimensions: {correlation_matrix.shape}")

    return correlation_matrix
