"""Functions for loading standard preference data."""

import pandas as pd
import json
import pathlib
from loguru import logger

REQUIRED_COLUMNS = ["text_a", "text_b"]


def switch_pref_labels_in_df(df):
    """
    Switches model_a and model_b labels.
    """

    def switch_preferred(row):
        win_col = "preferred_text"

        if row[win_col] == "text_a":
            return "text_b"
        elif row[win_col] == "text_b":
            return "text_a"
        else:
            return row[win_col]

    df["preferred_text"] = df.apply(switch_preferred, axis=1)
    return df


def load(
    path: str | pathlib.Path, switch_labels: bool = False, merge_prompts: bool = True
) -> pd.DataFrame:
    """
    Load the standard preference data from a CSV file.

    Args:
        path (str): The path to the CSV file.
    """

    path = pathlib.Path(path)

    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix == ".json":
        df = _load_from_ap_json(path)
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")

    # Check that the required columns are present
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset {path} is missing required columns: {missing_columns}"
        )

    if "prompt" in df.columns and merge_prompts:
        df = _add_prompt_to_texts(df)

    # switch original labels
    if switch_labels:
        df = switch_pref_labels_in_df(df)

    return df


def _load_from_ap_json(
    path: str,
    force_pref_text_validation: bool = False,
) -> pd.DataFrame:
    """
    Load the standard preference data from a AnnotatedPairs JSON file.
    """
    with open(path, "r", encoding="utf-8") as f:
        ap_data = json.load(f)

    default_annotator = ap_data["metadata"]["default_annotator"]

    ap_version = ap_data["metadata"]["version"]
    if ap_version in ["1.0"]:
        raise ValueError(
            f"Loading from AnnotatedPairs (AP) {ap_version} is currently not supported, only from AP v2.0 or higher."
        )

    # Transform from AnnotatedPairs JSON format to simple CSV dataframe
    rows = []
    for comparison in ap_data["comparisons"]:
        pref_text = comparison["annotations"].get(default_annotator, {}).get("pref")
        pref_text = pref_text.replace("a", "text_a").replace("b", "text_b")
        rows.append(
            {
                "id": comparison["id"],
                "prompt": comparison["prompt"],
                "text_a": comparison["response_a"]["text"],
                "text_b": comparison["response_b"]["text"],
                "model_a": comparison["response_a"].get("model", None),
                "model_b": comparison["response_b"].get("model", None),
                "preferred_text": pref_text,
            }
        )

    df = pd.DataFrame(rows)
    if force_pref_text_validation and df["preferred_text"].isna().all():
        raise ValueError(
            "No preferred text found during ICAI run, cannot generate principles (because no target)."
        )
    return df


def _add_prompt_to_texts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the prompt to the texts.
    """

    # sanity check to see if the prompt is always in the response
    # using 100 first rows
    prompt_in_response = df.iloc[:100].apply(
        lambda row: str(row["prompt"]) in str(row["text_a"])
        and str(row["prompt"]) in str(row["text_b"]),
        axis=1,
    )
    if prompt_in_response.all():
        logger.warning(
            "Prompt is always in the response. This is unexpected. "
            "This is likely because the prompt is already included in the text columns."
            "Skipping merging the prompt and text columns."
        )
        return df

    logger.info("Combining prompts and texts since separate prompt column provided.")

    df["_og_text_a"] = df["text_a"]
    df["_og_text_b"] = df["text_b"]

    # Parse the prompt and text together into
    # a chatbot conversation.
    for text_name in ["text_a", "text_b"]:
        df[text_name] = df[["prompt", text_name]].apply(
            lambda x: str(
                [
                    {"role": "user", "content": x["prompt"]},
                    {"role": "assistant", "content": x[text_name]},
                ]
            ),
            axis=1,
        )

    return df
