"""Convert the Sprouse, Schütze & Almeida (2013) Likert-scale release to a
long-format CSV of participant-level human acceptability ratings.

Input (all downloaded files; provenance in data/MANIFEST.yaml):
  data/raw/sprouse-li/SSA.data/LS experiment/LI.ls.results.csv
      Per-participant 7-point Likert ratings. Columns: participant, survey,
      order, judgment (raw 1-7), item, condition, zscores. The first five
      lines are citation/usage comments; the file is not UTF-8 (read latin-1).
  data/raw/sprouse-li/SSA.Materials.xlsx
      Item ID -> sentence text (primary text source; 300 conditions x 8
      lexicalizations).
  data/raw/morcela/linguistic_inquiry_data.csv
      Secondary text source: Tjuatja et al. (2025) release maps 1,450 of the
      same item IDs (case-folded) to sentence text. Used only where the
      materials spreadsheet ID does not match the results-file ID.

Output:
  data/human_ratings.csv with columns
      item_id        item code from the results file (e.g. 34.1.fox.26.*.03)
      participant_id "ls-<n>" (n = the source's participant number, unique
                     across the LS experiment; prefixed so ratings from other
                     tasks could be added later without collision)
      rating         raw Likert judgment, integer 1-7
      construction   the source's pairwise condition code (e.g. 34.1.fox.26.*,
                     i.e. LI volume.issue.author.example.judgment); no labels
                     were invented
      text           sentence text joined by item ID (case-insensitive).
                     EMPTY for the 120 items whose results-file ID matches
                     neither text source exactly (19 conditions have
                     divergent ID formats, e.g. results "32.2.nunes.3a.g" vs
                     materials "32.2.nunes.3a1.g"/"3a2.g"); left blank rather
                     than guessed.
      source         constant "sprouse_schutze_almeida_2013_lingua:LS"

Usage terms: the source data is posted for verification of the published
analyses; contact Jon Sprouse before using it for novel research (see the
header of the source CSV and data/MANIFEST.yaml).

Run from the repo root:  .venv/bin/python data/convert_human.py
"""

import warnings
from pathlib import Path

import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LS_RESULTS = ROOT / "data/raw/sprouse-li/SSA.data/LS experiment/LI.ls.results.csv"
MATERIALS = ROOT / "data/raw/sprouse-li/SSA.Materials.xlsx"
MORCELA = ROOT / "data/raw/morcela/linguistic_inquiry_data.csv"
OUT = ROOT / "data/human_ratings.csv"

SOURCE_LABEL = "sprouse_schutze_almeida_2013_lingua:LS"


def load_text_map() -> dict[str, str]:
    """Map lower-cased item ID -> sentence text.

    Primary: SSA.Materials.xlsx (columns A/B = bad ID/sentence, E/F = good
    ID/sentence; data starts at row 5). Secondary: the MORCELA CSV, which
    fixes a handful of typos but is only consulted for IDs the materials
    sheet does not contain.
    """
    text: dict[str, str] = {}

    with warnings.catch_warnings():  # openpyxl warns about an unknown extension
        warnings.simplefilter("ignore")
        wb = openpyxl.load_workbook(MATERIALS, read_only=True)
        ws = wb["Sheet1"]
        for row in ws.iter_rows(min_row=5, values_only=True):  # rows parse lazily
            for id_col, sent_col in ((0, 1), (4, 5)):
                if row[id_col] and row[sent_col]:
                    text[str(row[id_col]).strip().lower()] = str(row[sent_col]).strip()

    morcela = pd.read_csv(MORCELA)
    for _, r in morcela.iterrows():
        for id_col, sent_col in (("Bad ID", "Bad Sentence"), ("Good ID", "Good Sentence")):
            key = str(r[id_col]).strip().lower()
            if key not in text:  # materials sheet stays authoritative
                text[key] = str(r[sent_col]).strip()
    return text


def main() -> None:
    ls = pd.read_csv(LS_RESULTS, skiprows=5, encoding="latin-1")

    n_missing_rating = ls["judgment"].isna().sum()
    if n_missing_rating:
        print(f"dropping {n_missing_rating} rows with missing judgment")
        ls = ls.dropna(subset=["judgment"])

    ratings = ls["judgment"].astype(int)
    assert ratings.between(1, 7).all(), "Likert ratings outside 1-7"

    text_map = load_text_map()
    out = pd.DataFrame(
        {
            "item_id": ls["item"],
            "participant_id": "ls-" + ls["participant"].astype(str),
            "rating": ratings,
            "construction": ls["condition"],
            "text": ls["item"].str.lower().map(text_map).fillna(""),
            "source": SOURCE_LABEL,
        }
    ).sort_values(["item_id", "participant_id"], kind="stable")

    out.to_csv(OUT, index=False)

    no_text = out["text"] == ""
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  rows:                {len(out)}")
    print(f"  participants:        {out['participant_id'].nunique()}")
    print(f"  items:               {out['item_id'].nunique()}")
    print(f"  constructions:       {out['construction'].nunique()}")
    print(f"  rating range:        {out['rating'].min()}-{out['rating'].max()}")
    print(
        f"  items without text:  {out.loc[no_text, 'item_id'].nunique()} "
        f"({no_text.sum()} rows; ID-format mismatch, left blank rather than guessed)"
    )


if __name__ == "__main__":
    main()
