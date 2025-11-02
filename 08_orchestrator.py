#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
IN BED PREDICTION - PIPELINE ORCHESTRATOR
==============================================================================

Purpose: Execute complete end-to-end machine learning pipeline

This script automates the execution of all ML pipeline scripts with dependency
checking, logging, and error handling.

Usage:
    python 08_orchestrator.py              # interactive mode
    python 08_orchestrator.py --all        # run complete pipeline
    python 08_orchestrator.py --steps 1,3  # run specific steps
    python 08_orchestrator.py --clean      # clean outputs

This script:
1. Runs 01_exploratory_analysis.py (EDA and feature engineering)
2. Runs 02_preprocessing.py (data preparation)
3. Runs 03_train_models.py (model training)
4. Runs 04_evaluate_metrics.py (performance evaluation)
5. Runs 05_confusion_matrix.py (error analysis)
6. Runs 06_roc_curves.py (threshold-independent evaluation)
7. Runs 07_final_report.py (comprehensive documentation)
8. Provides error handling and progress tracking throughout pipeline

Author: Bruno Silva
Date: 2025
==============================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ==============================================================================

import os
import subprocess
import sys
import shutil
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict


# Try to import colorama for colored output
try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False

    class Fore:
        RED = GREEN = BLUE = YELLOW = CYAN = MAGENTA = RESET = ""

    class Style:
        BRIGHT = RESET_ALL = ""


# Configuration Constants
class Config:
    """Orchestrator configuration parameters."""
    # Pipeline scripts
    SCRIPTS = [
        "01_exploratory_analysis.py",
        "02_preprocessing.py",
        "03_train_models.py",
        "04_evaluate_metrics.py",
        "05_confusion_matrix.py",
        "06_roc_curves.py",
        "07_final_report.py"
    ]

    SCRIPT_NAMES = [
        "Exploratory Analysis",
        "Preprocessing",
        "Model Training",
        "Metrics Evaluation",
        "Confusion Matrix",
        "ROC Curves",
        "Final Report"
    ]

    # File and directory paths
    LOG_FILE = "execution.log"
    DATASET_FILE = "outputs/dataset.csv"
    PREDICTIONS_DIR = "outputs/predictions/"
    MODELS_DIR = "outputs/models/"
    METRICS_FILE = "outputs/comparative_metrics.csv"
    FINAL_REPORT = "outputs/FINAL_REPORT.md"

    # User interaction prompts
    PROMPT_OPTION = "\nOption: "
    PRESS_ENTER = "\nPress Enter to continue..."

    # Execution settings
    SCRIPT_TIMEOUT = 600  # 10 minutes

    # Required libraries
    REQUIRED_LIBRARIES = [
        'pandas', 'numpy', 'sklearn', 'matplotlib',
        'seaborn', 'pickle'
    ]

    # Expected outputs after each step
    EXPECTED_OUTPUTS: Dict[int, List[str]] = {
        0: [DATASET_FILE, "outputs/"],
        1: ["outputs/data_processed/X_train.pkl", "outputs/data_processed/y_train.pkl"],
        2: [MODELS_DIR, PREDICTIONS_DIR],
        3: [METRICS_FILE],
        4: ["outputs/confusion_matrix.png"],
        5: ["outputs/roc_curves.png"],
        6: [FINAL_REPORT]
    }


# ==============================================================================
# SECTION 2: LOGGING FUNCTIONS
# ==============================================================================


def log(message: str, level: str = "INFO") -> None:
    """
    Write message to log file with timestamp.

    Args:
        message: Message to log
        level: Log level (INFO, WARNING, ERROR, CRITICAL, SUCCESS)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {level}: {message}\n"

    with open(Config.LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_entry)


def print_colored(
        message: str,
        color: str = Fore.RESET,
        bright: bool = False) -> None:
    """
    Print colored message if colorama available.

    Args:
        message: Message to print
        color: Foreground color
        bright: Whether to use bright/bold style
    """
    if COLORS_AVAILABLE:
        style = Style.BRIGHT if bright else ""
        print(f"{style}{color}{message}{Style.RESET_ALL}")
    else:
        print(message)


def print_header(title: str) -> None:
    """
    Print formatted header.

    Args:
        title: Header title to display
    """
    print()
    print_colored("=" * 80, Fore.CYAN, bright=True)
    print_colored(title.center(80), Fore.CYAN, bright=True)
    print_colored("=" * 80, Fore.CYAN, bright=True)
    print()


def print_separator() -> None:
    """Print separator line."""
    print_colored("─" * 80, Fore.CYAN)


# ==============================================================================
# SECTION 3: DEPENDENCY CHECKING
# ==============================================================================


def check_dependencies() -> bool:
    """
    Check if required dependencies are installed.

    Returns:
        bool: True if all dependencies are installed, False otherwise
    """
    print_header("CHECKING DEPENDENCIES")
    log("Starting dependency check")

    missing = []

    for lib in Config.REQUIRED_LIBRARIES:
        try:
            __import__(lib)
            print_colored(f"  ✓ {lib}", Fore.GREEN)
        except ImportError:
            print_colored(f"  ✗ {lib} - NOT FOUND", Fore.RED)
            missing.append(lib)

    if missing:
        print()
        print_colored(
            f"[WARN]  Missing libraries: {
                ', '.join(missing)}",
            Fore.YELLOW,
            bright=True)
        print_colored(
            f"Install with: pip install {' '.join(missing)}", Fore.YELLOW)
        log(f"Missing libraries: {missing}", "WARNING")
        return False

    print()
    print_colored("✓ All dependencies installed", Fore.GREEN, bright=True)
    log("All dependencies OK")
    return True


def check_input_data() -> bool:
    """
    Check if input data exists.

    Returns:
        bool: True if data found or user chooses to continue, False to abort
    """
    print_header("CHECKING INPUT DATA")
    log("Checking input data")

    # Check for input folder
    input_dir = Path('inputs')
    data_dir = Path('data')

    if input_dir.exists():
        csv_files = list(input_dir.glob('*.csv'))
        if csv_files:
            print_colored(
                f"✓ Found {len(csv_files)} CSV files in inputs/", Fore.GREEN)
            log(f"Found {len(csv_files)} CSV files in inputs/")
            return True

    if data_dir.exists():
        csv_files = list(data_dir.glob('*.csv'))
        if csv_files:
            print_colored(
                f"✓ Found {len(csv_files)} CSV files in data/", Fore.GREEN)
            log(f"Found {len(csv_files)} CSV files in data/")
            return True

    # Data not found
    print_colored("✗ Input data not found", Fore.RED)
    print()
    print_colored("Dataset not found. Options:", Fore.YELLOW)
    print("  [1] Specify path to data folder")
    print("  [2] Continue anyway (maybe data is already processed)")
    print("  [3] Abort")

    choice = input(Config.PROMPT_OPTION).strip()

    if choice == '1':
        path = input("Enter path to data folder: ").strip()
        if Path(path).exists():
            print_colored(f"✓ Path found: {path}", Fore.GREEN)
            log(f"User specified data path: {path}")
            return True
        else:
            print_colored("✗ Path not found", Fore.RED)
            return False
    elif choice == '2':
        print_colored(
            "[WARN]  Continuing without input verification", Fore.YELLOW)
        log("Continued without input verification", "WARNING")
        return True
    else:
        print_colored("Execution aborted", Fore.RED)
        log("Execution aborted by user")
        sys.exit(0)


# ==============================================================================
# SECTION 4: SCRIPT EXECUTION
# ==============================================================================


def execute_script(script_path: str, script_name: str,
                   step_number: int) -> Tuple[bool, float, str, str]:
    """
    Execute a single script and return success status.

    Args:
        script_path: Path to the script file
        script_name: Human-readable name of the script
        step_number: Current step number

    Returns:
        Tuple[bool, float, str, str]: (success, elapsed_time, stdout, stderr)
    """
    print()
    print_separator()
    print_colored(f"▶ [{step_number}/{len(Config.SCRIPTS)}] Executing: {script_name}",
                  Fore.BLUE, bright=True)
    print_separator()

    log(f"Executing {script_path}")

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            env={**os.environ, "PYTHONUTF8": "1"},
            timeout=Config.SCRIPT_TIMEOUT
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print_colored(f"✓ Completed: {script_name} ({elapsed:.2f}s)",
                          Fore.GREEN, bright=True)
            log(f"{script_path} completed successfully ({elapsed:.2f}s)", "SUCCESS")
            return True, elapsed, result.stdout, result.stderr
        else:
            print_colored(f"✗ Error: {script_name}", Fore.RED, bright=True)
            log(f"{script_path} failed", "ERROR")
            log(f"Error details: {result.stderr}", "ERROR")
            return False, elapsed, result.stdout, result.stderr

    except subprocess.TimeoutExpired:
        print_colored(
            f"✗ Timeout: {script_name} (exceeded {
                Config.SCRIPT_TIMEOUT //
                60} minutes)",
            Fore.RED,
            bright=True)
        log(f"{script_path} timed out", "ERROR")
        return False, Config.SCRIPT_TIMEOUT, "", "Timeout exceeded"
    except Exception as e:
        print_colored(f"✗ Exception: {script_name} - {str(e)}",
                      Fore.RED, bright=True)
        log(f"{script_path} exception: {str(e)}", "ERROR")
        return False, 0, "", str(e)


def handle_script_error(script_name: str, stdout: str, stderr: str) -> str:
    """
    Handle script execution error.

    Args:
        script_name: Name of the failed script
        stdout: Standard output from the script
        stderr: Standard error from the script

    Returns:
        str: User's choice ('continue', 'retry', or 'abort')
    """
    print()
    print_colored("=" * 80, Fore.RED)
    print_colored(f"ERROR EXECUTING: {script_name}", Fore.RED, bright=True)
    print_colored("=" * 80, Fore.RED)

    print("\nWhat would you like to do?")
    print("  [1] Continue to next script (may cause cascade errors)")
    print("  [2] Retry this script")
    print("  [3] Abort execution")
    print("  [4] Debug mode (show full output)")

    choice = input(Config.PROMPT_OPTION).strip()

    if choice == '1':
        log("User chose to continue after error", "WARNING")
        return 'continue'
    elif choice == '2':
        log("User chose to retry script")
        return 'retry'
    elif choice == '4':
        print("\n" + "=" * 80)
        print("STDOUT:")
        print("=" * 80)
        print(stdout if stdout else "(empty)")
        print("\n" + "=" * 80)
        print("STDERR:")
        print("=" * 80)
        print(stderr if stderr else "(empty)")
        print("=" * 80)
        return handle_script_error(script_name, stdout, stderr)
    else:
        log("Execution aborted by user after error")
        return 'abort'


def verify_outputs(step_index: int) -> bool:
    """
    Verify expected outputs were created.

    Args:
        step_index: Index of the current step

    Returns:
        bool: True if all expected outputs exist, False otherwise
    """
    if step_index not in Config.EXPECTED_OUTPUTS:
        return True

    print()
    print_colored("Verifying outputs...", Fore.CYAN)

    all_exist = True
    for output in Config.EXPECTED_OUTPUTS[step_index]:
        path = Path(output)
        if path.exists():
            print_colored(f"  ✓ {output}", Fore.GREEN)
        else:
            print_colored(f"  [WARN]  {output} - NOT FOUND", Fore.YELLOW)
            log(f"Expected output not found: {output}", "WARNING")
            all_exist = False

    return all_exist


# ==============================================================================
# SECTION 5: PIPELINE EXECUTION
# ==============================================================================


def run_pipeline(steps_to_run: Optional[List[int]] = None) -> None:
    """
    Run the complete pipeline or specific steps.

    Args:
        steps_to_run: List of step indices to run, or None to run all steps
    """
    if steps_to_run is None:
        steps_to_run = list(range(len(Config.SCRIPTS)))

    print_header("STARTING PIPELINE EXECUTION")
    log("=" * 80)
    log("Pipeline execution started")
    log(f"Steps to run: {[i + 1 for i in steps_to_run]}")

    results = []
    total_time = 0

    for i in steps_to_run:
        script = Config.SCRIPTS[i]
        script_name = Config.SCRIPT_NAMES[i]

        # Execute script
        while True:
            success, elapsed, stdout, stderr = execute_script(
                script, script_name, i + 1
            )
            total_time += elapsed

            # Verify outputs
            if success:
                verify_outputs(i)
                results.append((script_name, True, elapsed))
                break
            else:
                results.append((script_name, False, elapsed))
                action = handle_script_error(script_name, stdout, stderr)

                if action == 'continue':
                    break
                elif action == 'retry':
                    print_colored("\nRetrying...", Fore.YELLOW)
                else:
                    print_colored("\nExecution aborted", Fore.RED)
                    log("Execution aborted by user")
                    print_summary(results, total_time, aborted=True)
                    sys.exit(1)

    print_summary(results, total_time)
    log("Pipeline execution completed")
    log(f"Total time: {total_time:.2f}s")


def print_summary(results: List[Tuple[str, bool, float]],
                  total_time: float, aborted: bool = False) -> None:
    """
    Print execution summary.

    Args:
        results: List of (script_name, success, elapsed_time) tuples
        total_time: Total execution time in seconds
        aborted: Whether execution was aborted
    """
    print()
    print_header("EXECUTION SUMMARY")

    successes = sum(1 for _, success, _ in results if success)
    failures = len(results) - successes

    if aborted:
        print_colored("[WARN]  EXECUTION ABORTED", Fore.YELLOW, bright=True)

    print(f"Scripts executed: {len(results)}/{len(Config.SCRIPTS)}")
    print_colored(f"Successes: {successes}", Fore.GREEN)
    if failures > 0:
        print_colored(f"Failures: {failures}", Fore.RED)

    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    print(f"\nTotal time: {minutes}m {seconds}s")

    print("\nDetailed results:")
    for script_name, success, elapsed in results:
        status = "✓" if success else "✗"
        color = Fore.GREEN if success else Fore.RED
        print_colored(f"  {status} {script_name} ({elapsed:.2f}s)", color)

    print("\nOutputs verification:")
    check_all_outputs()

    print(f"\nFull log: {Config.LOG_FILE}")
    print_separator()


def check_all_outputs() -> None:
    """Check all expected outputs."""
    outputs_to_check = [
        (Config.DATASET_FILE, "Consolidated dataset"),
        ("outputs/data_processed/", "Processed data"),
        (Config.MODELS_DIR, "Trained models"),
        (Config.PREDICTIONS_DIR, "Model predictions"),
        (Config.METRICS_FILE, "Metrics comparison"),
        ("outputs/confusion_matrix.png", "Confusion matrix"),
        ("outputs/roc_curves.png", "ROC curves"),
        (Config.FINAL_REPORT, "Final report")
    ]

    for path, description in outputs_to_check:
        if Path(path).exists():
            print_colored(f"  ✓ {description}", Fore.GREEN)
        else:
            print_colored(f"  ✗ {description}", Fore.RED)


# ==============================================================================
# SECTION 6: MENU AND USER INTERACTION
# ==============================================================================


def show_main_menu() -> str:
    """
    Show interactive main menu.

    Returns:
        str: User's menu selection
    """
    print_header(
        "ORCHESTRATOR - LAB 01.1\nClassification: Predict In Bed Probability")

    print("Select execution mode:")
    print("  [1] Run complete pipeline (all 7 scripts)")
    print("  [2] Run specific steps (choose which ones)")
    print("  [3] Resume execution (from last failed step)")
    print("  [4] Clean outputs and restart")
    print("  [5] Exit")

    return input(Config.PROMPT_OPTION).strip()


def select_specific_steps() -> Optional[List[int]]:
    """
    Allow user to select specific steps.

    Returns:
        Optional[List[int]]: List of selected step indices, or None if invalid
    """
    print_header("SELECT STEPS TO EXECUTE")

    for i, name in enumerate(Config.SCRIPT_NAMES, 1):
        print(f"  [{i}] {name}")

    print("\nEnter step numbers separated by commas (e.g., 1,3,4):")
    selection = input("Steps: ").strip()

    try:
        steps = [int(s.strip()) - 1 for s in selection.split(',')]
        steps = [s for s in steps if 0 <= s < len(Config.SCRIPTS)]

        if not steps:
            print_colored("Invalid selection", Fore.RED)
            return None

        print("\nSelected steps:")
        for i in steps:
            print(f"  • {Config.SCRIPT_NAMES[i]}")

        confirm = input("\nProceed? (y/n): ").strip().lower()
        if confirm == 'y':
            return steps

    except Exception:
        print_colored("Invalid input", Fore.RED)

    return None


def clean_outputs() -> None:
    """Clean all output files and folders."""
    print_header("CLEAN OUTPUTS")

    items_to_clean = [
        "outputs/",
        "data_processed/",
        Config.MODELS_DIR,
        Config.PREDICTIONS_DIR,
        Config.DATASET_FILE,
        Config.METRICS_FILE,
        "comparative_metrics.md",
        "auc_comparison.csv",
        "training_times.csv",
        Config.FINAL_REPORT,
        "FINAL_REPORT.pdf",
        Config.LOG_FILE
    ]

    print("The following items will be deleted:")
    for item in items_to_clean:
        if Path(item).exists():
            print_colored(f"  • {item}", Fore.YELLOW)

    print()
    confirm = input("ARE YOU SURE? (yes/no): ").strip().lower()

    if confirm == 'yes':
        for item in items_to_clean:
            path = Path(item)
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.is_file():
                    path.unlink()
            except Exception as e:
                print_colored(f"Error deleting {item}: {e}", Fore.RED)

        print_colored("\n✓ Cleanup completed", Fore.GREEN, bright=True)
        log("Outputs cleaned by user")
    else:
        print_colored("Cleanup cancelled", Fore.YELLOW)


# ==============================================================================
# SECTION 7: COMMAND LINE INTERFACE
# ==============================================================================


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Orchestrator for ML Pipeline - Lab 01.1"
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Run complete pipeline (non-interactive)'
    )

    parser.add_argument(
        '--steps',
        type=str,
        help='Run specific steps (comma-separated, e.g., 1,3,4)'
    )

    parser.add_argument(
        '--clean',
        action='store_true',
        help='Clean all outputs'
    )

    parser.add_argument(
        '--silent',
        action='store_true',
        help='Silent mode (no user interaction)'
    )

    return parser.parse_args()


# ==============================================================================
# SECTION 8: MAIN FUNCTION
# ==============================================================================


def main() -> None:
    """
    Main orchestrator function.

    Handles both interactive and command-line modes for running the ML pipeline.
    """
    args = parse_arguments()

    # Handle command line arguments
    if args.clean:
        clean_outputs()
        return

    if args.all or args.steps or args.silent:
        # Non-interactive mode
        log("Starting in non-interactive mode")

        if not check_dependencies():
            print_colored(
                "\n[WARN]  Missing dependencies. Aborting.", Fore.RED)
            sys.exit(1)

        if args.all:
            run_pipeline()
        elif args.steps:
            try:
                steps = [int(s.strip()) - 1 for s in args.steps.split(',')]
                run_pipeline(steps)
            except Exception:
                print_colored("Invalid steps format", Fore.RED)
                sys.exit(1)
        return

    # Interactive mode
    log("Starting in interactive mode")

    # Check dependencies
    if not check_dependencies():
        print()
        cont = input("Continue anyway? (y/n): ").strip().lower()
        if cont != 'y':
            sys.exit(1)

    # Check input data
    check_input_data()

    # Main menu loop
    while True:
        choice = show_main_menu()

        if choice == '1':
            run_pipeline()
            break
        elif choice == '2':
            steps = select_specific_steps()
            if steps:
                run_pipeline(steps)
            break
        elif choice == '3':
            print_colored("Resume mode not yet implemented", Fore.YELLOW)
            input(Config.PRESS_ENTER)
        elif choice == '4':
            clean_outputs()
            input(Config.PRESS_ENTER)
        elif choice == '5':
            print_colored("Goodbye!", Fore.CYAN)
            break
        else:
            print_colored("Invalid option", Fore.RED)
            input(Config.PRESS_ENTER)


# ==============================================================================
# EXECUTION
# ==============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print_colored("\n[WARN]  Execution interrupted by user", Fore.YELLOW)
        log("Execution interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print_colored(f"\n✗ Unexpected error: {e}", Fore.RED)
        log(f"Unexpected error: {e}", "CRITICAL")
        raise
