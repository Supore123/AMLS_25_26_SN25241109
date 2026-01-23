"""Reporting utilities for experiment results.

This module provides helpers to convert nested result dictionaries into
pretty pandas DataFrames and to persist them as CSV and LaTeX files.

Only functions are exposed and there is no top-level script execution.
All functionality is typed and uses logging rather than print statements.
"""

from typing import Dict
import logging
import os

import pandas as pd

# Module logger
logger = logging.getLogger(__name__)


def create_results_dataframe(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Create a pandas DataFrame from a nested results dictionary.

    Description:
        Converts a mapping of model names to metric dictionaries into a
        transposed DataFrame and applies sensible rounding for presentation.

    Args:
        results (Dict[str, Dict[str, float]]): Nested dictionary where the
            top-level keys are model names and values are metric-name -> value
            mappings.

    Returns:
        pd.DataFrame: Transposed DataFrame (models as rows) with values
            rounded to 4 decimal places for readability.
    """
    # Build DataFrame and transpose to have models as rows
    results_df: pd.DataFrame = pd.DataFrame(results).T

    # Round numeric values for nicer display in reports and tables
    results_df = results_df.round(4)
    return results_df


def save_results_files(
    results_df: pd.DataFrame,
    csv_path: str = os.path.join("Code", "output", "results", "results_summary.csv"),
    tex_path: str = os.path.join("Code", "output", "results", "results_table.tex"),
    latex_caption: str = "Comprehensive Model Comparison",
    latex_label: str = "tab:results",
) -> None:
    """
    Persist the results DataFrame to CSV and LaTeX files.

    Args:
        df (pd.DataFrame): DataFrame to save.
        csv_path (str): Output path for CSV file.
        tex_path (str): Output path for LaTeX file.
        latex_caption (str): Caption to include in the LaTeX table.
        latex_label (str): LaTeX label for referencing the table.

    Returns:
        None: Files are written to disk; errors are logged.
    """
    try:
        # Ensure output directories exist
        csv_dir = os.path.dirname(csv_path) or "."
        tex_dir = os.path.dirname(tex_path) or "."
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(tex_dir, exist_ok=True)

        # Save CSV
        results_df.to_csv(csv_path, index=True)
        logger.info("Saved results CSV to '%s'", csv_path)

        # Create LaTeX table. escape=False keeps formatting (e.g., percentage signs)
        latex = results_df.to_latex(caption=latex_caption, label=latex_label, escape=False)
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(latex)
        logger.info("Saved LaTeX table to '%s'", tex_path)
    except OSError as exc:  # pragma: no cover - IO errors surfaced here
        logger.exception("Failed to save result files due to IO error: %s", exc)


def generate_report(
    results: Dict[str, Dict[str, float]],
    csv_path: str = "results_summary.csv",
    tex_path: str = "results_table.tex",
) -> pd.DataFrame:
    """
    High-level helper to create a results DataFrame and persist it.

    This function composes the smaller helpers into a single, easy-to-call
    entry point that returns the DataFrame for further programmatic use.

    Args:
        results (Dict[str, Dict[str, float]]): Nested results mapping.
        csv_path (str): Path to write the CSV summary.
        tex_path (str): Path to write the LaTeX table.

    Returns:
        pd.DataFrame: The generated and rounded DataFrame.
    """
    results_df = create_results_dataframe(results)

    # Log a compact table summary at INFO level for visibility in pipelines
    logger.info("Generated results DataFrame with shape %s", results_df.shape)

    # Save artifacts for reports and LaTeX inclusion
    save_results_files(results_df, csv_path=csv_path, tex_path=tex_path)

    return results_df


def _example_results() -> Dict[str, Dict[str, float]]:
    """
    Produce a small example results dictionary used for local demonstration.

    Returns:
        Dict[str, Dict[str, float]]: Example nested results dictionary.
    """
    return {
        "SVM (Raw)": {
            "Accuracy": 0.85,
            "Precision": 0.83,
            "Recall": 0.87,
            "F1": 0.85,
            "AUC": 0.89,
            "Training Time (s)": 12.5,
        },
        "SVM (HOG)": {
            "Accuracy": 0.88,
            "Precision": 0.86,
            "Recall": 0.90,
            "F1": 0.88,
            "AUC": 0.92,
            "Training Time (s)": 18.3,
        },
        "ResNet (No Aug)": {
            "Accuracy": 0.86,
            "Precision": 0.84,
            "Recall": 0.88,
            "F1": 0.86,
            "AUC": 0.90,
            "Training Time (s)": 145.2,
        },
        "ResNet (Aug)": {
            "Accuracy": 0.90,
            "Precision": 0.89,
            "Recall": 0.91,
            "F1": 0.90,
            "AUC": 0.94,
            "Training Time (s)": 167.8,
        },
    }


if __name__ == "__main__":
    # Basic CLI-style demonstration guarded by the usual entry point.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.info("Running reporting module example to generate sample artifacts")

    example = _example_results()
    df = generate_report(example)
    logger.info("Example report generated. DataFrame shape: %s", df.shape)

