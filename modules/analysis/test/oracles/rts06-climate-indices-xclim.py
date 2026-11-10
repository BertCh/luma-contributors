# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Climate-index fixtures (RTS-06) executed with xclim 1.0.0 (Apache-2.0) and NumPy.

xclim needs numba and dask at import time; when they cannot be installed, the environment variable
RTS06_XCLIM_EXTRA_PATHS (os.pathsep-separated) may prepend directories that provide stand-ins. The
index code paths that this generator executes (rolling, group fit, scipy.stats CDFs) use neither
numba nor dask arrays; the fixture records what was stubbed. Output: fixtures/rts06-climate-indices-xclim.json.
"""
import json
import os
import pathlib
import platform
import sys
import warnings

for extra_path in reversed(os.environ.get('RTS06_XCLIM_EXTRA_PATHS', '').split(os.pathsep)):
    if extra_path:
        sys.path.insert(0, extra_path)

warnings.simplefilter('ignore')
import numpy as np
import pandas as pd
import scipy
import xarray as xr
import xclim
from xclim.compute import (
    growing_degree_days,
    heating_degree_days,
    maximum_consecutive_dry_days,
    maximum_consecutive_wet_days,
    standardized_precipitation_evapotranspiration_index,
    standardized_precipitation_index,
)

OUTPUT = pathlib.Path(__file__).resolve().parent / 'fixtures' / 'rts06-climate-indices-xclim.json'
CELL_COUNT = 2
MONTH_COUNT = 420  # 35 years: more slices than the 256-slice in-memory stack limit
CALIBRATION_START = '1990-01-01'
CALIBRATION_END = '2019-12-31'


def to_list(values):
    return [None if not np.isfinite(value) else float(value) for value in np.asarray(values, dtype=float).ravel()]


def monthly_series():
    months = pd.date_range('1990-01-01', periods=MONTH_COUNT, freq='MS')
    random = np.random.default_rng(20260109)
    precipitation = np.zeros((CELL_COUNT, MONTH_COUNT))
    water_balance = np.zeros((CELL_COUNT, MONTH_COUNT))
    for cell in range(CELL_COUNT):
        for index, month in enumerate(months):
            seasonal = 1.0 + 0.6 * np.sin(2 * np.pi * (month.month - 1) / 12 + cell)
            value = random.gamma(1.6, 3.0 * seasonal)
            if random.random() < 0.08 + 0.05 * cell:
                value = 0.0
            precipitation[cell, index] = value
            water_balance[cell, index] = value - 4.0 + 1.5 * np.cos(2 * np.pi * (month.month - 1) / 12)
    return months, precipitation, water_balance


def to_data_array(values, times, units):
    return xr.DataArray(values, dims=('cell', 'time'), coords={'time': times, 'cell': np.arange(values.shape[0])}, attrs={'units': units})


def daily_series():
    days = pd.date_range('2001-01-01', periods=1460, freq='D')
    random = np.random.default_rng(20260110)
    precipitation = np.zeros((CELL_COUNT, days.size))
    temperature = np.zeros((CELL_COUNT, days.size))
    for cell in range(CELL_COUNT):
        wet = False
        for index, day in enumerate(days):
            wet = random.random() < (0.6 if wet else 0.25)
            precipitation[cell, index] = random.gamma(1.2, 5.0) if wet else random.random() * 0.8
            temperature[cell, index] = 10.0 + 14.0 * np.sin(2 * np.pi * (day.dayofyear - 80) / 365.25) + random.normal(0, 3.0) + 2.0 * cell
    return days, precipitation, temperature


def main():
    months, precipitation, water_balance = monthly_series()
    monthly_precipitation = to_data_array(precipitation, months, 'mm/d')
    monthly_balance = to_data_array(water_balance, months, 'mm/d')
    window_cases = []
    for window in (1, 3):
        spi = standardized_precipitation_index(
            monthly_precipitation,
            freq='MS',
            window=window,
            dist='gamma',
            method='APP',
            fitkwargs={'floc': 0},
            cal_start=CALIBRATION_START,
            cal_end=CALIBRATION_END,
        ).transpose('cell', 'time')
        window_cases.append({'window': window, 'index': to_list(spi.values)})

    location = -30.0
    spei = standardized_precipitation_evapotranspiration_index(
        monthly_balance,
        freq='MS',
        window=1,
        dist='fisk',
        method='APP',
        fitkwargs={'floc': location},
        cal_start=CALIBRATION_START,
        cal_end=CALIBRATION_END,
    ).transpose('cell', 'time')

    days, precipitation_daily, temperature_daily = daily_series()
    daily_precipitation = to_data_array(precipitation_daily, days, 'mm/d')
    daily_temperature = to_data_array(temperature_daily, days, 'degC')
    wet_threshold = 1.0
    wet_days = maximum_consecutive_wet_days(daily_precipitation, thresh=f'{wet_threshold} mm/d', freq='YS').transpose('cell', 'time')
    dry_days = maximum_consecutive_dry_days(daily_precipitation, thresh=f'{wet_threshold} mm/d', freq='YS').transpose('cell', 'time')
    growing_threshold = 4.0
    heating_threshold = 17.0
    growing = growing_degree_days(daily_temperature, thresh=f'{growing_threshold} degC', freq='YS').transpose('cell', 'time')
    heating = heating_degree_days(daily_temperature, thresh=f'{heating_threshold} degC', freq='YS').transpose('cell', 'time')

    monthly_group = np.array([month.month - 1 for month in months])
    climatology_cases = []
    for cell in range(CELL_COUNT):
        for slot in range(12):
            selected = precipitation[cell][monthly_group == slot]
            climatology_cases.append({'cell': cell, 'slot': slot, 'count': int(selected.size), 'mean': float(selected.mean()), 'sampleVariance': float(selected.var(ddof=1))})

    fixture = {
        'source': 'xclim.compute (standardized indices, threshold indices), NumPy for climatology',
        'license': 'Apache-2.0 (xclim 1.0.0)',
        'versions': {
            'python': platform.python_version(),
            'xclim': xclim.__version__,
            'numpy': np.__version__,
            'scipy': scipy.__version__,
            'pandas': pd.__version__,
            'xarray': xr.__version__,
        },
        'stubbedImports': [name for name in ('numba', 'dask') if getattr(sys.modules.get(name), '__file__', '') and 'rts06-oracle' in sys.modules[name].__file__],
        'cellCount': CELL_COUNT,
        'monthCount': MONTH_COUNT,
        'calibrationMonthCount': 360,
        'monthlySlots': monthly_group.tolist(),
        'precipitation': to_list(precipitation),
        'waterBalance': to_list(water_balance),
        'waterBalanceLocation': location,
        'windows': window_cases,
        'spei': to_list(spei.values),
        'dayCount': int(days.size),
        'yearLengths': [int(count) for count in pd.Series(days.year).value_counts().sort_index().tolist()],
        'precipitationDaily': to_list(precipitation_daily),
        'temperatureDaily': to_list(temperature_daily),
        'wetThreshold': wet_threshold,
        'growingThreshold': growing_threshold,
        'heatingThreshold': heating_threshold,
        'maximumConsecutiveWetDays': to_list(wet_days.values),
        'maximumConsecutiveDryDays': to_list(dry_days.values),
        'growingDegreeDays': to_list(growing.values),
        'heatingDegreeDays': to_list(heating.values),
        'climatology': climatology_cases,
    }
    OUTPUT.write_text(json.dumps(fixture))
    print('wrote', OUTPUT, OUTPUT.stat().st_size)


if __name__ == '__main__':
    main()
