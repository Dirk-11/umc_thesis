"""
01_data_prep.py — Parse the MOCKI Excel and produce clean labeled CSVs.

Pipeline:
  1. Read the Excel sheet
  2. Extract per-photo rows (drop lookup-table rows at bottom)
  3. Forward-fill Fragment IDs (only set on first photo of each stone)
  4. Strip whitespace from component columns (handles 'CHP ' typo)
  5. Exclude stones listed in config (S168, S301)
  6. Apply known percentage corrections (Excel data-entry errors)
  7. Apply OTH remapping based on FTIR sub-identification
  8. Apply purity threshold → binary pure/mixed label
  9. Build composition vectors (comp_CaOx, comp_CHP, …) for Model B
 10. Verify images exist on disk for each stone
 11. Export stone-level and image-level CSVs

Outputs:
  - data/processed/stones_labeled.csv   (one row per stone)
  - data/processed/images_labeled.csv   (one row per image, joined with stone labels)

Run:
  python scripts/01_data_prep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent))
from utils import ensure_dir, load_config, resolve_path, setup_logging

log = setup_logging("data_prep")


# -----------------------------------------------------------------------------
# Excel parsing
# -----------------------------------------------------------------------------
def load_raw_excel(excel_path: Path) -> pd.DataFrame:
    """Read the Excel file and return rows that correspond to actual photos."""
    log.info(f"Reading {excel_path}")
    df = pd.read_excel(excel_path)

    # Rename obscure column headers
    df = df.rename(columns={
        "Unnamed: 3": "pure_mixed_raw",
        "Analyse primair": "primary",
        "%": "primary_pct",
        "Analyse secundair": "secondary",
        "%.1": "secondary_pct",
        "Analyse tertiair": "tertiary",
        "%.2": "tertiary_pct",
        "Analyse quartiair": "quaternary",
        "%.3": "quaternary_pct",
        "Unnamed: 18": "oth_detail",
        "Fragment": "fragment",
        "Foto": "photo",
    })

    # Keep only rows that have a photo ID (drops lookup table at the bottom)
    df = df[df["photo"].notna()].copy()

    # Forward-fill fragment IDs (set only on first photo of each stone)
    df["fragment"] = df["fragment"].ffill()

    # Strip whitespace from component columns (handles 'CHP ' with trailing space)
    for col in ["primary", "secondary", "tertiary"]:
        df[col] = (
            df[col].astype(str).str.strip()
            .replace({"nan": np.nan, "": np.nan})
        )

    log.info(f"  {len(df)} photo rows, {df['fragment'].nunique()} unique stones")
    return df


# -----------------------------------------------------------------------------
# Percentage corrections (known Excel data-entry errors)
# -----------------------------------------------------------------------------
def apply_percentage_corrections(df: pd.DataFrame, corrections: dict) -> pd.DataFrame:
    """Fix known percentage column-swap errors and typos in the raw Excel data.

    Corrections are defined in config.yaml under data.percentage_corrections so
    the pipeline is reproducible even when re-run from the raw Excel.

    Args:
        df:          Per-photo DataFrame (all 3 rows of each stone are corrected).
        corrections: Dict mapping fragment ID → {column: correct_value}.
    """
    if not corrections:
        return df
    df = df.copy()
    for fragment, fixes in corrections.items():
        mask = df["fragment"] == fragment
        if not mask.any():
            log.warning(f"  Correction target '{fragment}' not found in data — skipping")
            continue
        for col, val in fixes.items():
            df.loc[mask, col] = val
        log.info(f"  Corrected {fragment}: {fixes}")
    return df


# -----------------------------------------------------------------------------
# Class remapping (two-stage: OTH subtypes → classes, then component merging)
# -----------------------------------------------------------------------------
def apply_class_remapping(df: pd.DataFrame, remap_cfg: dict) -> pd.DataFrame:
    """Apply two-stage class remapping.

    Stage 1 — OTH subtype remapping:
      When a component cell is 'OTH' and the FTIR sub-identification
      (`oth_detail` column) matches a known mineral, remap to a final class
      (e.g. 'Na-H-urate hydrate' → 'UA'). Anything not matched stays 'OTH'.

    Stage 2 — Standard component merging:
      Collapse the 9 original Excel component codes into the 6 final
      classes (e.g. COM + COD → CaOx, CHP + CHPD + CMP → CHP).

    Both stages operate on primary, secondary, AND tertiary columns so
    downstream regression/composition models get consistent labels.
    """
    if not remap_cfg["enabled"]:
        log.info("Class remapping disabled in config — keeping original labels")
        return df

    df = df.copy()
    spelling_map = remap_cfg["oth_spelling_normalization"]
    oth_to_final = remap_cfg["oth_to_final_class"]
    comp_to_final = remap_cfg["component_to_final_class"]

    # Normalize OTH detail spelling
    df["oth_normalized"] = df["oth_detail"].map(spelling_map)

    # --- Log stage 1 remapping (stone-level counts, to avoid 3x inflation) ---
    stone_view = df.drop_duplicates(subset="fragment")
    remappable = stone_view[stone_view["oth_normalized"].notna()].copy()
    remappable["target"] = remappable["oth_normalized"].map(oth_to_final).fillna("OTH")
    stage1 = remappable[remappable["target"] != "OTH"]["target"].value_counts()
    if len(stage1) > 0:
        log.info("Stage 1 — OTH subtype remapping:")
        for target, n in stage1.items():
            log.info(f"  {n} OTH stones → {target}")

    # --- Apply stage 1: OTH → final class (where we have a rule) ---
    for position in ["primary", "secondary", "tertiary"]:
        mask = (df[position] == "OTH") & df["oth_normalized"].notna()
        for idx in df[mask].index:
            target = oth_to_final.get(df.at[idx, "oth_normalized"])
            if target is not None:
                df.at[idx, position] = target

    # --- Apply stage 2: component code → final class ---
    log.info("Stage 2 — component merging:")
    for src, dst in comp_to_final.items():
        if src != dst:
            log.info(f"  {src} → {dst}")

    for position in ["primary", "secondary", "tertiary"]:
        df[position] = df[position].map(lambda c: comp_to_final.get(c, c) if pd.notna(c) else c)

    return df


# -----------------------------------------------------------------------------
# Stone-level aggregation
# -----------------------------------------------------------------------------
def build_stone_table(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-photo rows into per-stone rows.

    All 3 photos of a stone share the same composition, so we just take
    the first photo's composition fields.
    """
    # Composition is identical across the 3 photos of a stone
    stone_df = df.drop_duplicates(subset="fragment", keep="first").copy()
    stone_df = stone_df[[
        "fragment", "primary", "primary_pct",
        "secondary", "secondary_pct",
        "tertiary", "tertiary_pct",
        "pure_mixed_raw", "oth_detail", "oth_normalized",
    ]].reset_index(drop=True)
    return stone_df


def apply_purity_threshold(stones: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Add a binary pure/mixed label based on the primary component percentage."""
    stones = stones.copy()
    stones["purity_threshold"] = threshold
    stones["label"] = np.where(
        stones["primary_pct"] >= threshold, "pure", "mixed"
    )
    # Stones with missing primary_pct get no label
    stones.loc[stones["primary_pct"].isna(), "label"] = None
    return stones


# -----------------------------------------------------------------------------
# Composition vectors for Model B (compositional regression)
# -----------------------------------------------------------------------------
def build_composition_vectors(stones: pd.DataFrame, final_classes: list[str]) -> pd.DataFrame:
    """Add comp_<class> columns (float, sum to 1.0) to the stone-level DataFrame.

    Aggregates primary/secondary/tertiary percentages into per-class buckets
    (a stone may have COM=60% and COD=30%, both mapping to CaOx → 90%), then
    normalises each stone's vector to sum exactly to 1.0.

    Stones with no valid composition data get all-zero vectors.
    """
    stones = stones.copy()
    comp_cols = [f"comp_{cls}" for cls in final_classes]

    # Initialise all class columns to 0
    for col in comp_cols:
        stones[col] = 0.0

    # Accumulate percentages into class buckets (vectorised over classes)
    for pos, pct_col in [
        ("primary",   "primary_pct"),
        ("secondary", "secondary_pct"),
        ("tertiary",  "tertiary_pct"),
    ]:
        for cls in final_classes:
            mask = stones[pos] == cls
            stones.loc[mask, f"comp_{cls}"] += stones.loc[mask, pct_col].fillna(0.0)

    # Normalise each row to sum to 1.0
    totals = stones[comp_cols].sum(axis=1)
    valid = totals > 0
    stones.loc[valid, comp_cols] = stones.loc[valid, comp_cols].div(totals[valid], axis=0)

    # Sanity check
    bad = (stones[comp_cols].sum(axis=1) - 1.0).abs().gt(0.01) & valid
    if bad.any():
        log.warning(f"  {bad.sum()} stones still don't sum to 1.0 after normalisation")
    log.info(f"  Built composition vectors for {valid.sum()} stones "
             f"({(~valid).sum()} with no composition data)")

    return stones


# -----------------------------------------------------------------------------
# Image file verification
# -----------------------------------------------------------------------------
def verify_images(
    df: pd.DataFrame, images_dir: Path, extension: str
) -> pd.DataFrame:
    """Check each photo has a corresponding image file on disk.

    Adds columns:
      image_path: absolute path to the image (None if missing)
      image_exists: True/False
    """
    df = df.copy()
    df["image_path"] = df["photo"].apply(
        lambda name: str(images_dir / f"{name}{extension}") if pd.notna(name) else None
    )
    df["image_exists"] = df["image_path"].apply(
        lambda p: Path(p).exists() if p else False
    )

    n_missing = (~df["image_exists"]).sum()
    n_total = len(df)
    log.info(f"  Image files found: {n_total - n_missing}/{n_total}")
    if n_missing > 0:
        log.warning(f"  {n_missing} images missing — expected if not all uploaded yet")

    return df


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    cfg = load_config()
    excel_path = resolve_path(cfg, "excel_file")
    images_dir = resolve_path(cfg, "images_dir")

    # 1. Load and clean
    df = load_raw_excel(excel_path)

    # 2. Exclude problem stones
    exclude = cfg["data"]["exclude_stones"]
    before = df["fragment"].nunique()
    df = df[~df["fragment"].isin(exclude)].copy()
    after = df["fragment"].nunique()
    log.info(f"Excluded {before - after} stones: {exclude}")

    # 3. Apply known Excel percentage corrections
    corrections = cfg["data"].get("percentage_corrections", {})
    if corrections:
        log.info("Applying percentage corrections:")
        df = apply_percentage_corrections(df, corrections)

    # 4. Apply class remapping (OTH subtypes + standard merging)
    df = apply_class_remapping(df, cfg["class_remapping"])

    # 4. Build stone-level table
    stones = build_stone_table(df)

    # 5. Apply purity threshold
    threshold = cfg["data"]["purity_threshold"]
    stones = apply_purity_threshold(stones, threshold)
    log.info(f"Applied purity threshold: >={threshold}%")
    log.info(f"  Pure:  {(stones['label'] == 'pure').sum()}")
    log.info(f"  Mixed: {(stones['label'] == 'mixed').sum()}")
    log.info(f"  Unlabeled (missing data): {stones['label'].isna().sum()}")

    # 6. Component distribution after remapping
    log.info("Final class distribution (after remapping):")
    for comp, n in stones["primary"].value_counts().items():
        log.info(f"  {comp}: {n}")

    # 7. Build composition vectors for Model B
    final_classes = cfg["class_remapping"]["final_classes"]
    log.info(f"Building composition vectors for classes: {final_classes}")
    stones = build_composition_vectors(stones, final_classes)

    # 8. Verify images
    ext = cfg["data"]["image_extension"]
    df = verify_images(df, images_dir, ext)

    # 9. Build image-level table (one row per photo, joined with stone label)
    comp_cols = [f"comp_{cls}" for cls in final_classes]
    images = df[[
        "fragment", "photo", "image_path", "image_exists",
        "primary", "primary_pct",
    ]].merge(
        stones[["fragment", "label", "purity_threshold"] + comp_cols],
        on="fragment",
    )

    # 9. Save outputs
    processed_dir = ensure_dir(
        resolve_path(cfg, "processed_csv").parent
    )
    stones_out = processed_dir / "stones_labeled.csv"
    images_out = processed_dir / "images_labeled.csv"
    stones.to_csv(stones_out, index=False)
    images.to_csv(images_out, index=False)

    log.info(f"Saved: {stones_out}")
    log.info(f"Saved: {images_out}")
    log.info("Done.")


if __name__ == "__main__":
    main()
