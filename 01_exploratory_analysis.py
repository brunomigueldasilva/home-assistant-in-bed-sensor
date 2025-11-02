#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
IN BED PREDICTION - EXPLORATORY DATA ANALYSIS
==============================================================================

Purpose: Analyze sensor data to predict when person is lying in bed

This script:
1. Loads 7 sensor CSV files (lights, blinds, motion, TV) + sleep tracking
2. Creates target variable 'in_bed' based on sleep windows
3. Consolidates data and engineers temporal features (30-min routines)
4. Analyzes class imbalance and feature correlations
5. Creates 7 comprehensive visualizations
6. Saves consolidated dataset ready for preprocessing

Author: Bruno Silva
Date: 2025
==============================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ==============================================================================

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')


# Configuration Constants
class Config:
    """Exploratory analysis configuration parameters."""
    # Directories
    INPUT_DIR = Path('inputs')
    OUTPUT_DIR = Path('outputs')
    GRAPHICS_DIR = OUTPUT_DIR / 'graphics'

    # Output files
    DATASET_CSV = OUTPUT_DIR / 'dataset.csv'

    # Visualization settings
    DPI = 300
    FIGSIZE = (12, 6)
    FONT_SIZE = 10

    # Column names constants
    DATE_OF_SLEEP = "SleepScore 4 semanas"
    BED_TIME = 'Hora de deitar'
    WAKE_TIME = 'Hora a que acordou'

    # Class labels
    IN_BED = 'In Bed'
    NOT_IN_BED = 'Not In Bed'

    # Sensor files
    SENSOR_FILES = {
        'bedroom_blinds': 'bedroom_blinds.csv',
        'hallway_light': 'hallway_light.csv',
        'bedroom_light': 'bedoroom_light.csv',
        'wc_light': 'wc_light.csv',
        'bedroom_tv': 'bedroom_tv.csv',
        'motion': 'hallway_motion_sensor.csv'
    }
    SLEEP_FILE = 'sleep.csv'

    # Feature engineering
    TIME_WINDOW_MINUTES = 30


# Visualization configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = Config.FIGSIZE
plt.rcParams['font.size'] = Config.FONT_SIZE


# ==============================================================================
# SECTION 2: UTILITY FUNCTIONS
# ==============================================================================


def save_plot(filename: str, dpi: int = 300) -> None:
    """Save current plot to outputs folder."""
    filepath = Path(Config.GRAPHICS_DIR) / filename
    plt.tight_layout()
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"  → Plot saved: {filepath}")
    plt.close()


def print_section(title: str, char: str = "=") -> None:
    """Print formatted section header."""
    print("\n" + char * 80)
    print(title)
    print(char * 80)


# ==============================================================================
# SECTION 3: DATA LOADING FUNCTIONS
# ==============================================================================


def load_sensor_csv(filepath: Path,
                    sensor_name: str) -> Optional[pd.DataFrame]:
    """
    Load a sensor CSV file and convert last_changed to datetime.

    Args:
        filepath: Path to CSV file
        sensor_name: Descriptive sensor name (for logging)

    Returns:
        DataFrame with sensor data or None if error occurs
    """
    try:
        df = pd.read_csv(filepath)
        original_records = len(df)

        # Convert to datetime and remove timezone to avoid conflicts
        df['last_changed'] = pd.to_datetime(
            df['last_changed'], errors='coerce')
        if df['last_changed'].dt.tz is not None:
            df['last_changed'] = df['last_changed'].dt.tz_localize(None)

        # Remove records with 'unavailable' or 'unknown' state
        invalid_states = ['unavailable', 'unknown', 'none', '']
        df = df[~df['state'].astype(str).str.lower().isin(invalid_states)]

        # Remove records with invalid timestamp
        df = df.dropna(subset=['last_changed'])

        records_after_cleaning = len(df)
        removed_records = original_records - records_after_cleaning

        # Add sensor identifier column
        df['sensor'] = sensor_name

        print(f"✓ {sensor_name}: {records_after_cleaning} valid records "
              f"({removed_records} removed: unavailable/unknown)")

        return df
    except Exception as e:
        print(f"✗ Error loading {sensor_name}: {e}")
        return None


def load_sleep_data(filepath: Path) -> Optional[pd.DataFrame]:
    """
    Load and process sleep data file.

    Args:
        filepath: Path to Sleep.csv

    Returns:
        DataFrame with processed sleep data or None if error occurs
    """
    try:
        if not filepath.exists():
            print(f"✗ File not found: {filepath}")
            return None

        sleep_df = pd.read_csv(filepath)
        print(f"✓ Sleep.csv: {len(sleep_df)} records loaded")
        print(f"  Original columns: {list(sleep_df.columns)}")

        # Keep only the necessary columns
        columns_to_keep = [
            Config.DATE_OF_SLEEP,
            Config.BED_TIME,
            Config.WAKE_TIME]
        sleep_df = sleep_df[columns_to_keep].copy()

        # Convert date column
        sleep_df[Config.DATE_OF_SLEEP] = pd.to_datetime(
            sleep_df[Config.DATE_OF_SLEEP], errors='coerce'
        ).dt.date

        # Convert time columns
        sleep_df[Config.BED_TIME] = pd.to_datetime(
            sleep_df[Config.BED_TIME], errors='coerce').dt.time
        sleep_df[Config.WAKE_TIME] = pd.to_datetime(
            sleep_df[Config.WAKE_TIME], errors='coerce').dt.time

        # Combine date + time into full datetime
        sleep_df[Config.BED_TIME] = sleep_df.apply(
            lambda r: pd.Timestamp.combine(r[Config.DATE_OF_SLEEP], r[Config.BED_TIME])
            if pd.notnull(r[Config.BED_TIME]) and pd.notnull(r[Config.DATE_OF_SLEEP])
            else pd.NaT,
            axis=1
        )

        sleep_df[Config.WAKE_TIME] = sleep_df.apply(
            lambda r: pd.Timestamp.combine(r[Config.DATE_OF_SLEEP], r[Config.WAKE_TIME])
            if pd.notnull(r[Config.WAKE_TIME]) and pd.notnull(r[Config.DATE_OF_SLEEP])
            else pd.NaT,
            axis=1
        )

        print(f"  Columns kept: {list(sleep_df.columns)}")
        print(f"  Valid sleep windows: {sleep_df.dropna().shape[0]}")

        return sleep_df

    except Exception as e:
        print(f"✗ Error loading Sleep.csv: {e}")
        return None


def load_all_data() -> Tuple[Dict[str, pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load all necessary CSV files.

    Returns:
        Tuple of (sensor_data dict, sleep_df)
    """
    print_section("1. DATA LOADING")

    # Check input folder
    if not Config.INPUT_DIR.exists():
        print(
            f"✗ ERROR: Input folder not found: {
                Config.INPUT_DIR.absolute()}")
        print(
            f"Please create the '{
                Config.INPUT_DIR}' folder and place CSV files there.")
        raise FileNotFoundError(
            f"Input directory not found: {
                Config.INPUT_DIR}")

    # List available CSV files
    print(f"Searching for CSV files in: {Config.INPUT_DIR.absolute()}")
    csv_files = list(Config.INPUT_DIR.glob('*.csv'))
    print(f"CSV files found: {len(csv_files)}")
    for f in csv_files:
        print(f"  - {f.name}")

    if not csv_files:
        print("\n⚠️  WARNING: No CSV files found!")
        print("Please place CSV files in the input folder.")
        raise FileNotFoundError("No CSV files found in input directory")

    # Define sensor file paths
    files = {
        key: Config.INPUT_DIR / filename
        for key, filename in Config.SENSOR_FILES.items()
    }

    # Load sensor files
    sensor_data = {}
    for key, filepath in files.items():
        df = load_sensor_csv(filepath, key)
        if df is not None:
            sensor_data[key] = df

    # Load sleep file
    print("\nLoading sleep data...")
    sleep_df = load_sleep_data(Config.INPUT_DIR / Config.SLEEP_FILE)

    return sensor_data, sleep_df


# ==============================================================================
# SECTION 4: TARGET VARIABLE CREATION FUNCTIONS
# ==============================================================================


def normalize_timestamp(timestamp: pd.Timestamp) -> pd.Timestamp:
    """Remove timezone from timestamp if present."""
    if hasattr(timestamp, 'tz') and timestamp.tz is not None:
        return timestamp.tz_localize(None)
    return timestamp


def check_in_bed(timestamp: pd.Timestamp, sleep_windows: pd.DataFrame) -> int:
    """
    Check if a timestamp is within any sleep window.

    Logic: A timestamp is "in bed" if it falls between "bedtime" and "wake time"
    from any record in the Sleep.csv file.

    Args:
        timestamp: Timestamp to check
        sleep_windows: DataFrame with BED_TIME and WAKE_TIME columns

    Returns:
        1 if in bed, 0 otherwise
    """
    if pd.isna(timestamp):
        return 0

    # Normalize timezone
    timestamp = normalize_timestamp(timestamp)

    for _, window in sleep_windows.iterrows():
        start = window[Config.BED_TIME]
        end = window[Config.WAKE_TIME]

        # Ignore windows with missing values
        if pd.isna(start) or pd.isna(end):
            continue

        # Normalize timezone of windows
        start = normalize_timestamp(start)
        end = normalize_timestamp(end)

        # Check if timestamp is within window
        if start <= timestamp <= end:
            return 1

    return 0


def create_target_variable(
    sensor_data: Dict[str, pd.DataFrame],
    sleep_df: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """
    Create target variable 'in_bed' by consolidating sensor data.

    Args:
        sensor_data: Dictionary with sensor DataFrames
        sleep_df: DataFrame with sleep windows

    Returns:
        Consolidated DataFrame with target variable
    """
    print_section("2. TARGET VARIABLE CREATION: 'in_bed'")

    print("""
TARGET VARIABLE LOGIC:
- For each timestamp in sensor data, we check if that moment is within
  any sleep window recorded in Sleep.csv
- Sleep window = period between "bedtime" and "wake time"
- in_bed = 1: timestamp within a sleep window
- in_bed = 0: timestamp outside any sleep window
""")

    # Consolidate all sensor timestamps
    print("Consolidating timestamps from all sensors...")
    all_timestamps = []

    for sensor_name, df in sensor_data.items():
        df_temp = df[['last_changed', 'state', 'sensor']].copy()
        df_temp.rename(columns={'last_changed': 'timestamp'}, inplace=True)
        all_timestamps.append(df_temp)

    # Check if there's data to consolidate
    if not all_timestamps:
        raise ValueError("No sensor data was loaded successfully!")

    # Create consolidated DataFrame
    df_consolidated = pd.concat(all_timestamps, ignore_index=True)
    df_consolidated = df_consolidated.dropna(subset=['timestamp'])
    df_consolidated = df_consolidated.sort_values(
        'timestamp').reset_index(drop=True)

    print(f"  Total records: {len(df_consolidated)}")
    print(f"  Period: {df_consolidated['timestamp'].min()} to "
          f"{df_consolidated['timestamp'].max()}")

    # Create target variable
    print("\nCreating target variable 'in_bed'...")
    if sleep_df is not None:
        df_consolidated['in_bed'] = df_consolidated['timestamp'].apply(
            lambda x: check_in_bed(x, sleep_df)
        )
        print("  ✓ Target variable 'in_bed' created successfully")
    else:
        print("  ✗ Cannot create target variable without sleep data")
        df_consolidated['in_bed'] = 0

    return df_consolidated


# ==============================================================================
# SECTION 5: DATA CONSOLIDATION AND BASIC FEATURES
# ==============================================================================


def consolidate_data(
    df_consolidated: pd.DataFrame,
    sleep_df: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """
    Consolidate data in wide format and add basic temporal features.

    Args:
        df_consolidated: DataFrame with consolidated data
        sleep_df: DataFrame with sleep data

    Returns:
        Pivoted DataFrame with features
    """
    print_section("3. DATA CONSOLIDATION")

    print("""
CONSOLIDATION STRATEGY:
- Transform data from "long" format (multiple rows per sensor) to "wide"
- Each row represents a unique timestamp
- Each column represents a sensor's state at that moment
- 'unavailable' records were removed during loading
- For timestamps where a sensor has no record, we use forward fill (last valid state)
""")

    print("\nPivoting data to wide format...")
    df_pivot = df_consolidated.pivot_table(
        index='timestamp',
        columns='sensor',
        values='state',
        aggfunc='first'
    ).reset_index()

    # Fill missing values with forward fill (propagate last valid state)
    print("  Filling missing values (forward fill - last valid state)...")
    df_pivot = df_pivot.ffill()

    # If there are still NaN at the beginning, use backward fill
    df_pivot = df_pivot.bfill()

    # Check if there are still missing values
    columns_with_nan = df_pivot.columns[df_pivot.isna().any()].tolist()
    if columns_with_nan:
        print(
            f"  ⚠️  Warning: Columns still with missing values: {columns_with_nan}")
        print("     Filling with 'off' (default state)...")
        df_pivot = df_pivot.fillna('off')

    # Add target variable
    print("  Adding target variable...")
    df_pivot['in_bed'] = df_pivot['timestamp'].apply(
        lambda x: check_in_bed(x, sleep_df) if sleep_df is not None else 0
    )

    # Add temporal features
    print("  Extracting temporal features...")
    df_pivot['hour'] = df_pivot['timestamp'].dt.hour
    df_pivot['minute'] = df_pivot['timestamp'].dt.minute
    df_pivot['day_of_week'] = df_pivot['timestamp'].dt.dayofweek
    df_pivot['is_weekend'] = (df_pivot['day_of_week'] >= 5).astype(int)
    df_pivot['date'] = df_pivot['timestamp'].dt.date

    print("\n✓ Dataset consolidated:")
    print(f"  Dimensions: {df_pivot.shape}")
    print(f"  Columns: {list(df_pivot.columns)}")
    print("\n  Unique values per sensor:")

    # Show unique values of each sensor for validation
    temporal_cols = ['timestamp', 'in_bed', 'hour', 'minute', 'day_of_week',
                     'is_weekend', 'date']
    for col in df_pivot.columns:
        if col not in temporal_cols:
            unique_values = df_pivot[col].unique()
            print(f"    {col}: {unique_values}")

    return df_pivot


# ==============================================================================
# SECTION 6: FEATURE ENGINEERING FUNCTIONS
# ==============================================================================


def event_in_time_window(
    df: pd.DataFrame,
    timestamp: pd.Timestamp,
    sensor_column: str,
    positive_values: List[str],
    window_minutes: int = 30
) -> int:
    """
    Check if a specific sensor event occurred in the time window before timestamp.

    Args:
        df: Complete DataFrame
        timestamp: Reference timestamp
        sensor_column: Sensor column name
        positive_values: List of values indicating the event
        window_minutes: Time window size in minutes

    Returns:
        1 if event occurred, 0 otherwise
    """
    lower_limit = timestamp - pd.Timedelta(minutes=window_minutes)
    mask = (df['timestamp'] > lower_limit) & (df['timestamp'] <= timestamp)
    window_records = df[mask]

    if sensor_column in window_records.columns:
        events = window_records[sensor_column].isin(positive_values)
        return 1 if events.any() else 0
    return 0


def create_temporal_features(
        df_pivot: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create features based on events in the last 30 minutes.

    Args:
        df_pivot: Consolidated DataFrame

    Returns:
        Tuple of (DataFrame with new features, list of feature names)
    """
    print_section("3.5. FEATURE ENGINEERING - TEMPORAL FEATURES CREATION")

    print("""
TEMPORAL FEATURES LOGIC:
Capture bedtime routine patterns based on recent events (last 30 min):

1. hallway_motion_30min: Motion sensor detected presence
2. wc_light_on_30min: WC light was on
3. bedroom_light_on_30min: Bedroom light was on
4. partial_routine_30min: Motion + WC (partial routine)
5. complete_routine_30min: Motion + WC + Bedroom (complete routine)
""")

    print("Creating features based on events in the last 30 minutes...")
    df_pivot = df_pivot.sort_values('timestamp').reset_index(drop=True)

    # Feature 1: Hallway motion (check all motion sensors)
    print("  → Creating 'hallway_motion_30min'...")
    motion_columns = [col for col in df_pivot.columns
                      if 'motion' in col.lower()]

    if motion_columns:
        print(f"     Found motion sensors: {motion_columns}")
        df_pivot['hallway_motion_30min'] = df_pivot['timestamp'].apply(
            lambda x: max([
                event_in_time_window(
                    df_pivot, x, col, ['on', 'detected', 'motion'], 30
                )
                for col in motion_columns
            ])
        )
    else:
        print("     ⚠️ No motion sensors found, setting to 0")
        df_pivot['hallway_motion_30min'] = 0

    # Feature 2: WC light on (check both left and right)
    print("  → Creating 'wc_light_on_30min'...")
    wc_light_columns = [
        col for col in df_pivot.columns
        if col in ['wc_light', 'wc_light_left', 'wc_light_right']
    ]

    if wc_light_columns:
        print(f"     Found WC light sensors: {wc_light_columns}")
        # Check if ANY WC light was on in the last 30 minutes
        df_pivot['wc_light_on_30min'] = df_pivot['timestamp'].apply(
            lambda x: max([event_in_time_window(df_pivot, x, col, ['on'], 30)
                          for col in wc_light_columns])
        )
    else:
        print("     ⚠️ No WC light sensors found, setting to 0")
        df_pivot['wc_light_on_30min'] = 0

    # Feature 3: Bedroom light on (check all bedroom lights)
    print("  → Creating 'bedroom_light_on_30min'...")
    bedroom_light_columns = [col for col in df_pivot.columns
                             if 'bedroom_light' in col]

    if bedroom_light_columns:
        print(f"     Found bedroom light sensors: {bedroom_light_columns}")
        df_pivot['bedroom_light_on_30min'] = df_pivot['timestamp'].apply(
            lambda x: max([event_in_time_window(df_pivot, x, col, ['on'], 30)
                          for col in bedroom_light_columns])
        )
    else:
        print("     ⚠️ No bedroom light sensors found, setting to 0")
        df_pivot['bedroom_light_on_30min'] = 0

    # Feature 4: Partial routine (motion + WC)
    print("  → Creating 'partial_routine_30min'...")
    df_pivot['partial_routine_30min'] = (
        (df_pivot['hallway_motion_30min'] == 1) &
        (df_pivot['wc_light_on_30min'] == 1)
    ).astype(int)

    # Feature 5: Complete routine
    print("  → Creating 'complete_routine_30min'...")
    df_pivot['complete_routine_30min'] = (
        (df_pivot['hallway_motion_30min'] == 1) &
        (df_pivot['wc_light_on_30min'] == 1) &
        (df_pivot['bedroom_light_on_30min'] == 1)
    ).astype(int)

    temporal_features = [
        'hallway_motion_30min',
        'wc_light_on_30min',
        'bedroom_light_on_30min',
        'partial_routine_30min',
        'complete_routine_30min'
    ]

    # Statistics
    print("\n✓ Temporal features created successfully!")
    print("\nNew features statistics:")
    for feature in temporal_features:
        count = df_pivot[feature].value_counts()
        pct_positive = (count.get(1, 0) / len(df_pivot)) * 100
        print(
            f"  {feature}: {
                count.get(
                    1,
                    0)} positive events ({
                pct_positive:.2f}%)")

    # Correlations
    print("\nCorrelation of temporal features with 'in_bed':")
    for feature in temporal_features:
        corr = df_pivot[[feature, 'in_bed']].corr().iloc[0, 1]
        print(f"  {feature}: {corr:.4f}")

    return df_pivot, temporal_features


# ==============================================================================
# SECTION 7: ANALYSIS AND VISUALIZATION FUNCTIONS
# ==============================================================================


def analyze_imbalance(df_pivot: pd.DataFrame) -> None:
    """Analyze and visualize target variable imbalance."""
    print_section("4. TARGET VARIABLE IMBALANCE ANALYSIS")

    counts = df_pivot['in_bed'].value_counts().sort_index()
    percentages = df_pivot['in_bed'].value_counts(
        normalize=True).sort_index() * 100

    print("\nTarget variable distribution:")
    print(f"  Not in bed (0): {counts.get(0, 0)} records "
          f"({percentages.get(0, 0):.2f}%)")
    print(f"  In bed (1): {counts.get(1, 0)} records "
          f"({percentages.get(1, 0):.2f}%)")

    # Check if both classes exist
    if len(counts) < 2:
        print("\n  ⚠️ WARNING: Only one class found in the data!")
        print("  This will cause issues with model training.")
        print("  Please ensure sleep tracking data is present and correctly processed.")
        return

    ratio = max(percentages) / min(percentages)
    print(f"\n  Imbalance ratio: {ratio:.2f}:1")

    print("""
IMBALANCE DISCUSSION:
- Ratio ~2:1 is considered MODERATE (not severe)
- Accuracy can be misleading
- Important metrics: Precision, Recall, F1-Score, ROC-AUC
- Consider class weights in models
""")

    # Plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        [Config.NOT_IN_BED, Config.IN_BED],
        counts.values,
        color=['#3498db', '#e74c3c'],
        alpha=0.7,
        edgecolor='black'
    )

    for bar, pct in zip(bars, percentages.values):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{pct:.1f}%\n({int(height)} records)',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )

    plt.title(
        'Target Variable Distribution: In Bed vs Not in Bed',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    plt.ylabel('Number of Records', fontsize=12)
    plt.xlabel('Class', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    save_plot('01_class_imbalance.png')


def analyze_categorical_features(
    df_pivot: pd.DataFrame,
    sensor_columns: List[str]
) -> None:
    """Analyze and visualize categorical features (sensors)."""
    print_section("5. CATEGORICAL FEATURES ANALYSIS (SENSORS)")

    print(f"\nSensors identified: {sensor_columns}")

    for sensor in sensor_columns:
        print_section(f"SENSOR: {sensor}", "─")

        unique_values = df_pivot[sensor].unique()
        print(f"  Unique values: {unique_values}")

        cross_tab = pd.crosstab(
            df_pivot[sensor],
            df_pivot['in_bed'],
            normalize='columns'
        ) * 100
        print("\n  Distribution by class (%):")
        print(cross_tab.round(2))

        # Plot
        _, axes = plt.subplots(1, 2, figsize=(14, 5))

        sensor_count = df_pivot[sensor].value_counts()
        axes[0].bar(
            range(len(sensor_count)),
            sensor_count.values,
            color='steelblue',
            alpha=0.7,
            edgecolor='black'
        )
        axes[0].set_xticks(range(len(sensor_count)))
        axes[0].set_xticklabels(sensor_count.index, rotation=45, ha='right')
        axes[0].set_title(f'{sensor} - Total Distribution', fontweight='bold')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(axis='y', alpha=0.3)

        cross_tab_counts = pd.crosstab(df_pivot[sensor], df_pivot['in_bed'])
        cross_tab_counts.plot(
            kind='bar',
            ax=axes[1],
            color=['#3498db', '#e74c3c'],
            alpha=0.7,
            edgecolor='black'
        )
        axes[1].set_title(f'{sensor} - By Class', fontweight='bold')
        axes[1].set_xlabel('')
        axes[1].set_ylabel('Frequency')
        axes[1].legend([Config.NOT_IN_BED, Config.IN_BED], loc='best')
        axes[1].set_xticklabels(
            axes[1].get_xticklabels(),
            rotation=45,
            ha='right')
        axes[1].grid(axis='y', alpha=0.3)

        plt.suptitle(
            f'Sensor Analysis: {sensor}',
            fontsize=14,
            fontweight='bold',
            y=1.02
        )
        save_plot(f'02_sensor_{sensor.replace(" ", "_")}.png')


def analyze_basic_temporal_features(df_pivot: pd.DataFrame) -> None:
    """Analyze and visualize basic temporal features (hour, day of week)."""
    print_section("6. BASIC TEMPORAL FEATURES ANALYSIS")

    _, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    # Hour distribution
    axes[0].hist(
        df_pivot['hour'],
        bins=24,
        color='teal',
        alpha=0.7,
        edgecolor='black')
    axes[0].set_title('Records Distribution by Hour', fontweight='bold')
    axes[0].set_xlabel('Hour of Day')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(axis='y', alpha=0.3)

    # Distribution by class
    for in_bed_val, label, color in [(0, Config.NOT_IN_BED, '#3498db'),
                                     (1, Config.IN_BED, '#e74c3c')]:
        data = df_pivot[df_pivot['in_bed'] == in_bed_val]['hour']
        axes[1].hist(
            data,
            bins=24,
            alpha=0.6,
            label=label,
            color=color,
            edgecolor='black'
        )

    axes[1].set_title('Hour Distribution by Class', fontweight='bold')
    axes[1].set_xlabel('Hour of Day')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    # Day of week
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    day_counts = df_pivot['day_of_week'].value_counts().sort_index()
    axes[2].bar(
        range(7),
        day_counts.values,
        color='coral',
        alpha=0.7,
        edgecolor='black'
    )
    axes[2].set_xticks(range(7))
    axes[2].set_xticklabels(day_names)
    axes[2].set_title('Distribution by Day of Week', fontweight='bold')
    axes[2].set_ylabel('Frequency')
    axes[2].grid(axis='y', alpha=0.3)

    # Weekend vs weekday
    weekend_counts = df_pivot.groupby(
        ['is_weekend', 'in_bed']).size().unstack()
    weekend_counts.plot(
        kind='bar',
        ax=axes[3],
        color=['#3498db', '#e74c3c'],
        alpha=0.7,
        edgecolor='black'
    )
    axes[3].set_title('Distribution: Weekend vs Weekday', fontweight='bold')
    axes[3].set_xticklabels(['Weekday', 'Weekend'], rotation=0)
    axes[3].set_ylabel('Frequency')
    axes[3].legend([Config.NOT_IN_BED, Config.IN_BED])
    axes[3].grid(axis='y', alpha=0.3)

    plt.suptitle(
        'Basic Temporal Features Analysis',
        fontsize=16,
        fontweight='bold',
        y=1.00
    )
    save_plot('03_basic_temporal_features.png')


def visualize_temporal_features(
    df_pivot: pd.DataFrame,
    temporal_features: List[str]
) -> None:
    """Visualize created temporal features (last 30 min)."""
    print("\nGenerating temporal features visualizations...")

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, feature in enumerate(temporal_features):
        cross_tab = pd.crosstab(
            df_pivot[feature],
            df_pivot['in_bed'],
            normalize='columns'
        ) * 100
        cross_tab_counts = pd.crosstab(df_pivot[feature], df_pivot['in_bed'])
        cross_tab_counts.plot(
            kind='bar',
            ax=axes[idx],
            color=['#3498db', '#e74c3c'],
            alpha=0.7,
            edgecolor='black'
        )

        # Calculate lift
        if (1 in cross_tab.index and 1 in cross_tab.columns and
                0 in cross_tab.columns):
            lift = (cross_tab.loc[1, 1] / cross_tab.loc[1, 0]
                    if cross_tab.loc[1, 0] > 0 else 0)
            axes[idx].set_title(
                f'{feature}\n(Lift: {lift:.2f}x when present)',
                fontweight='bold',
                fontsize=10
            )
        else:
            axes[idx].set_title(f'{feature}', fontweight='bold', fontsize=10)

        axes[idx].set_xlabel('Feature Present')
        axes[idx].set_ylabel('Frequency')
        axes[idx].legend([Config.NOT_IN_BED, Config.IN_BED], loc='best')

        # Set x-tick labels based on actual data
        actual_ticks = cross_tab_counts.index.tolist()
        tick_labels = ['No' if x == 0 else 'Yes' for x in actual_ticks]
        axes[idx].set_xticks(range(len(actual_ticks)))
        axes[idx].set_xticklabels(tick_labels, rotation=0)

        axes[idx].grid(axis='y', alpha=0.3)

    if len(temporal_features) < 6:
        fig.delaxes(axes[5])

    plt.suptitle(
        'Temporal Features (30 min) vs Target Variable',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )
    save_plot('03.5_temporal_features_30min.png')

    # Complete routine analysis
    if df_pivot['complete_routine_30min'].sum() > 0:
        print_section("COMPLETE ROUTINE ANALYSIS", "─")

        cross_routine = pd.crosstab(
            df_pivot['complete_routine_30min'],
            df_pivot['in_bed']
        )
        cross_routine_pct = pd.crosstab(
            df_pivot['complete_routine_30min'],
            df_pivot['in_bed'],
            normalize='index'
        ) * 100

        print("\nCounts:")
        print(cross_routine)
        print("\nPercentages (by row):")
        print(cross_routine_pct.round(2))

        if 1 in cross_routine_pct.index and 1 in cross_routine_pct.columns:
            prob_in_bed_with_routine = cross_routine_pct.loc[1, 1]
            prob_in_bed_without_routine = (
                cross_routine_pct.loc[0, 1] if 0 in cross_routine_pct.index else 0
            )
            print(
                "\n📊 INSIGHT: When complete routine is present, probability of "
                "being in bed")
            print(f"    is {prob_in_bed_with_routine:.1f}% vs "
                  f"{prob_in_bed_without_routine:.1f}% when absent")
            print(
                f"    → Increase of " f"{
                    prob_in_bed_with_routine -
                    prob_in_bed_without_routine:.1f} " f"percentage points!")


def analyze_correlations(
    df_pivot: pd.DataFrame,
    sensor_columns: List[str],
    temporal_features: List[str]
) -> None:
    """Analyze and visualize correlations between features."""
    print_section("7. CORRELATION ANALYSIS")

    print("\nCreating basic encoding for correlation analysis...")
    df_corr = df_pivot.copy()

    # Label encoding for categorical variables
    for col in sensor_columns:
        if df_corr[col].dtype == 'object':
            le = LabelEncoder()
            df_corr[col] = le.fit_transform(df_corr[col].astype(str))

    # Select numeric columns
    numeric_columns = (
        ['hour', 'minute', 'day_of_week', 'is_weekend', 'in_bed'] +
        sensor_columns +
        temporal_features
    )
    df_corr_num = df_corr[numeric_columns]

    # Calculate correlation matrix
    corr_matrix = df_corr_num.corr()

    print("\nCorrelations with target variable 'in_bed':")
    corr_with_target = corr_matrix['in_bed'].sort_values(ascending=False)
    print(corr_with_target)

    # Heatmap
    plt.figure(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title(
        'Correlation Matrix - Features vs Target Variable',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    save_plot('04_correlation_matrix.png')

    # Bar plot of correlations
    plt.figure(figsize=(12, 6))
    corr_with_target_sorted = corr_with_target.drop('in_bed').sort_values()
    colors = ['red' if x < 0 else 'green' for x in corr_with_target_sorted.values]
    plt.barh(
        range(len(corr_with_target_sorted)),
        corr_with_target_sorted.values,
        color=colors,
        alpha=0.7,
        edgecolor='black'
    )
    plt.yticks(range(len(corr_with_target_sorted)),
               corr_with_target_sorted.index)
    plt.xlabel('Pearson Correlation', fontsize=12)
    plt.title(
        'Feature Correlations with Target Variable "in_bed"',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    plt.grid(axis='x', alpha=0.3)
    save_plot('05_correlations_with_target.png')


def get_sensor_columns(df_pivot: pd.DataFrame,
                       temporal_features: List[str]) -> List[str]:
    """Extract sensor column names from dataframe."""
    excluded_cols = (['timestamp',
                      'in_bed',
                      'hour',
                      'minute',
                      'day_of_week',
                      'is_weekend',
                      'date'] + temporal_features)
    return [col for col in df_pivot.columns if col not in excluded_cols]


def print_summary(df_pivot: pd.DataFrame, counts: pd.Series) -> None:
    """Print final analysis summary."""
    print_section("EXPLORATORY ANALYSIS SUMMARY")

    # Safe ratio calculation
    pct = df_pivot['in_bed'].value_counts(normalize=True) * 100
    ratio = max(pct) / min(pct) if len(pct) >= 2 else 0

    in_bed_count = counts.get(1, 0)
    not_in_bed_count = counts.get(0, 0)

    imbalance_status = (
        "⚠️ ONLY ONE CLASS - DATA ISSUE"
        if len(counts) < 2
        else f"{ratio:.2f}:1 (MODERATE)"
    )

    sensor_count = len(get_sensor_columns(
        df_pivot,
        ['hallway_motion_30min', 'wc_light_on_30min', 'bedroom_light_on_30min',
         'partial_routine_30min', 'complete_routine_30min']
    ))

    print(f"""
✓ Consolidated dataset: {df_pivot.shape[0]} records, {df_pivot.shape[1]} features
✓ Target variable created: {in_bed_count} in bed records, {not_in_bed_count} not in bed
✓ Imbalance: {imbalance_status}
✓ Categorical features: {sensor_count} sensors analyzed
✓ Basic temporal features: hour, day_of_week, is_weekend
✓ Advanced temporal features (30 min): 5 features created
✓ Correlations calculated and visualized

Files generated:
  - dataset.csv (consolidated dataset)
  - outputs/01_class_imbalance.png
  - outputs/02_sensor_*.png (one per sensor)
  - outputs/03_basic_temporal_features.png
  - outputs/03.5_temporal_features_30min.png
  - outputs/04_correlation_matrix.png
  - outputs/05_correlations_with_target.png

RECOMMENDED NEXT STEPS:
1. Feature selection based on identified correlations
2. Experiment with different class balancing techniques
3. Train classification models (Logistic Regression, Random Forest, XGBoost)
4. Evaluate models with appropriate metrics (F1-Score, ROC-AUC, PR-AUC)
5. Perform hyperparameter tuning and threshold optimization
""")

    print("=" * 80)
    print("EXPLORATORY ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


# ==============================================================================
# SECTION 8: MAIN FUNCTION
# ==============================================================================


def main() -> None:
    """Main function that coordinates the entire exploratory analysis."""
    # Create output folder
    Config.GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"✓ Output folder created/verified: {Config.GRAPHICS_DIR.absolute()}")

    print("=" * 80)
    print("EXPLORATORY ANALYSIS - CLASSIFICATION: AM I LYING IN BED?")
    print("=" * 80)

    try:
        # 1. Load data
        sensor_data, sleep_df = load_all_data()

        # 2. Create target variable
        df_consolidated = create_target_variable(sensor_data, sleep_df)

        # 3. Consolidate data
        df_pivot = consolidate_data(df_consolidated, sleep_df)

        # 4. Feature Engineering
        df_pivot, temporal_features = create_temporal_features(df_pivot)

        # 5. Imbalance analysis
        analyze_imbalance(df_pivot)

        # 6. Categorical features analysis
        sensor_columns = get_sensor_columns(df_pivot, temporal_features)
        analyze_categorical_features(df_pivot, sensor_columns)

        # 7. Basic temporal features analysis
        analyze_basic_temporal_features(df_pivot)

        # 8. Visualize advanced temporal features
        visualize_temporal_features(df_pivot, temporal_features)

        # 9. Correlation analysis
        analyze_correlations(df_pivot, sensor_columns, temporal_features)

        # 10. Save final dataset
        df_pivot.to_csv(Config.DATASET_CSV, index=False)
        print(f"\n✓ Final dataset saved: {Config.DATASET_CSV}")

        # 11. Summary
        counts = df_pivot['in_bed'].value_counts().sort_index()
        print_summary(df_pivot, counts)

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        raise


# ==============================================================================
# EXECUTION
# ==============================================================================

if __name__ == "__main__":
    main()
