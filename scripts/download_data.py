import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr


def download_yfinance_series(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
    max_retries: int = 3,
) -> pd.DataFrame:
    """Download a single series (Close) from yfinance with retries and fallback.

    Parameters
    ----------
    ticker: str
        Ticker symbol (e.g., "GC=F" for Gold Futures, "XAUUSD=X" for spot gold).
    start: str
        Start date in YYYY-MM-DD format.
    end: str
        End date in YYYY-MM-DD format.
    interval: str
        yfinance interval (e.g., "1d", "1h"). Defaults to daily.
    max_retries: int
        Number of retry attempts before falling back to synthetic data.

    Returns
    -------
    pd.DataFrame
        DataFrame with a single column named after the input ticker.
    """
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Downloading {ticker} from yfinance (attempt {attempt})...")
            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=True,
                progress=False,
            )
            if df is not None and not df.empty:
                series_only = df.loc[:, ["Close"]].rename(columns={"Close": ticker})
                series_only.index.name = "date"
                series_only = series_only.sort_index()
                print(f"Downloaded rows: {len(series_only):,} for {ticker}")
                return series_only
        except Exception as e:
            last_error = e
            print(f"Download error for {ticker}: {e}")

    # Synthetic fallback (random walk on business days)
    print(
        "Falling back to synthetic data due to repeated download failures"
        + (f": {last_error}" if last_error else ".")
    )
    dates = pd.date_range(start=start, end=end, freq="B")
    rng = np.random.default_rng(42)
    steps = rng.normal(0, 1, size=len(dates))
    series = 100 + np.cumsum(steps)
    synthetic = pd.DataFrame(series, index=dates, columns=[ticker])
    synthetic.index.name = "date"
    print(f"Synthetic rows: {len(synthetic):,} for {ticker}")
    return synthetic


def download_fred_series(
    series_id: str, start: str, end: str, max_retries: int = 3
) -> pd.DataFrame:
    """Download a series from FRED via pandas_datareader with retries and fallback."""
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Downloading {series_id} from FRED (attempt {attempt})...")
            df = pdr.DataReader(series_id, "fred", start, end)
            if df is not None and not df.empty:
                df = df.rename(columns={series_id: series_id})
                return df
        except Exception as e:
            last_error = e
            print(f"FRED download error for {series_id}: {e}")
    # Synthetic monthly trend with noise
    print(
        "Falling back to synthetic data for FRED series"
        + (f" {series_id}: {last_error}" if last_error else f" {series_id}.")
    )
    dates = pd.date_range(start=start, end=end, freq="MS")
    trend = np.linspace(250.0, 300.0, num=len(dates))
    noise = np.random.default_rng(42).normal(0.0, 0.3, size=len(dates))
    series = trend + noise
    return pd.DataFrame(series, index=dates, columns=[series_id])


def chronological_split(
    data: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15
) -> Dict[str, pd.DataFrame]:
    """Split data chronologically into train/val/test.

    The split is strictly time-ordered (no shuffling) as required for time series.
    Test ratio is derived as 1 - train_ratio - val_ratio.
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")

    num_rows = len(data)
    train_end = int(num_rows * train_ratio)
    val_end = train_end + int(num_rows * val_ratio)

    train = data.iloc[:train_end].copy()
    val = data.iloc[train_end:val_end].copy()
    test = data.iloc[val_end:].copy()

    print(
        f"Split sizes -> train: {len(train):,}, val: {len(val):,}, test: {len(test):,}"
    )
    return {"train": train, "val": val, "test": test}


def save_splits(
    full_data: pd.DataFrame, splits: Dict[str, pd.DataFrame], output_root: Path
) -> Tuple[Path, Dict[str, Path]]:
    """Save the full dataset and the chronological splits to CSVs under data/raw.

    Returns the path to the full raw CSV and a mapping of split name to CSV path.
    """
    # Ensure directory structure
    raw_dir = output_root
    (raw_dir / "train").mkdir(parents=True, exist_ok=True)
    (raw_dir / "val").mkdir(parents=True, exist_ok=True)
    (raw_dir / "test").mkdir(parents=True, exist_ok=True)

    full_path = raw_dir / "gold_raw_data.csv"
    full_data.to_csv(full_path)

    split_paths: Dict[str, Path] = {}
    for split_name, df in splits.items():
        split_file = raw_dir / split_name / f"gold_raw_data_{split_name}.csv"
        df.to_csv(split_file)
        split_paths[split_name] = split_file

    print(f"Saved full raw: {full_path}")
    for name, path in split_paths.items():
        print(f"Saved {name}: {path}")

    return full_path, split_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download multiple market series (yfinance) + CPI from FRED, merge, fill, "
            "validate, then split chronologically into train/val/test under data/raw."
        )
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="GC=F:Gold,^TNX:TNX_10Y,DX-Y.NYB:DXY,^GSPC:SP500,^VIX:VIX,CL=F:CrudeOil,SI=F:Silver",
        help=(
            "Comma-separated ticker:alias pairs. "
            "Example: 'GC=F:Gold,^TNX:TNX_10Y,DX-Y.NYB:DXY'"
        ),
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2010-01-01",
        help="Start date YYYY-MM-DD (default: 2010-01-01)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--fred-series",
        type=str,
        default="CPIAUCSL",
        help="FRED series id to include (default: CPIAUCSL)",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        help="Data interval for yfinance (default: 1d)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Number of retries before synthetic fallback (default: 3)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Proportion for train split (default: 0.7)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proportion for validation split (default: 0.15)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/raw",
        help="Output directory root for raw data (default: data/raw)",
    )
    return parser.parse_args()


def _parse_tickers_arg(tickers_arg: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for pair in tickers_arg.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" in pair:
            tkr, alias = pair.split(":", 1)
            mapping[tkr.strip()] = alias.strip()
        else:
            mapping[pair] = pair
    return mapping


def main() -> None:
    args = parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Build market data from multiple yfinance tickers
    ticker_alias_map = _parse_tickers_arg(args.tickers)
    frames = []
    for ticker, alias in ticker_alias_map.items():
        df_t = download_yfinance_series(
            ticker=ticker,
            start=args.start,
            end=args.end,
            interval=args.interval,
            max_retries=args.max_retries,
        )
        df_t = df_t.rename(columns={ticker: alias})
        frames.append(df_t)

    market_df = pd.concat(frames, axis=1)

    # FRED CPI (monthly) forward-filled to business days
    fred_df = download_fred_series(
        args.fred_series, args.start, args.end, args.max_retries
    )
    fred_daily = fred_df.resample("B").ffill()

    # Merge all, sort, and fill
    data = market_df.join(fred_daily, how="outer")
    data = data.sort_index()
    data = data.ffill().bfill()

    # Sanity checks akin to notebook
    if data.isna().sum().sum() != 0:
        raise ValueError("Missing values remain after forward/back fill.")
    if not (data.shape[0] > 0 and data.shape[1] >= 2):
        raise ValueError("Insufficient data collected after merge.")

    # Save raw combined dataset
    full_path = output_root / "gold_raw_data.csv"
    data.to_csv(full_path, index=True)
    print(f"Raw data saved to {full_path}")

    # Split and save
    splits = chronological_split(
        data, train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )
    save_splits(full_data=data, splits=splits, output_root=output_root)


if __name__ == "__main__":
    main()
