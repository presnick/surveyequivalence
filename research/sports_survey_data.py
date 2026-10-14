"""Verify and prepare the open NBA individual-forecast release; no scoring.

Source: Rutchick et al. (2020), OSF Study 3, https://osf.io/2rzcs/.
Only reads the downloaded original workbook. Never places bets or accesses
accounts. Forecast/game records are written into the ignored local cache.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache/sports-survey-data"
OUTPUT = ROOT / "benchmarks/results/sports-data-search"
EXPECTED = "a900bd24143e5bf6692df807a76c9b28bbccaf52031c101b7cc37e7ccd7c2d4f"
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def read_workbook(path):
    with ZipFile(path) as archive:
        strings = ["".join(n.itertext()) for n in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheets = workbook.findall("s:sheets/s:sheet", NS)
        assert len(sheets) == 1
        xml = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in xml.findall("s:sheetData/s:row", NS):
        cells = {}
        for cell in row.findall("s:c", NS):
            value = cell.find("s:v", NS)
            if value is not None:
                cells[cell.attrib["r"].rstrip("0123456789")] = (
                    strings[int(value.text)] if cell.get("t") == "s" else float(value.text))
        rows.append((int(row.attrib["r"]), cells))
    return sheets[0].attrib["name"], rows


def prepare(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EXPECTED:
        raise ValueError("Workbook differs from the checked OSF release")
    sheet, rows = read_workbook(path)
    header = rows[0][1]
    columns = {label: col for col, label in header.items()}
    assert len(columns) == len(header)
    people = [(i, row) for i, row in rows[1:] if isinstance(row.get("A"), (int, float))]
    ids = [int(row["A"]) for _, row in people]
    assert len(ids) == len(set(ids)) == 130
    winners = [row for _, row in rows if str(row.get("A")).lower() == "winner"]
    assert len(winners) == 1
    winner_row = winners[0]
    game_ids = [label[:-5] for label in columns if label.endswith("_PRED")]
    assert len(set(game_ids)) == len(game_ids) == 465
    records, coverage = [], []
    raw_active, valid_active = set(), set()
    invalid, missing = Counter(), Counter()
    for game_id in game_ids:
        block = {kind: columns[game_id + "_" + kind] for kind in ["PRED", "CONF", "OTHERS"]}
        winner = winner_row[block["PRED"]]
        assert winner in (1, 2)
        n_picks = n_complete = 0
        for row_num, row in people:
            pid = int(row["A"])
            pick, confidence, others = (row.get(block[kind]) for kind in ["PRED", "CONF", "OTHERS"])
            if pick in (1, 2):
                n_picks += 1
                raw_active.add(pid)
            for kind, value in [("PRED", pick), ("CONF", confidence), ("OTHERS", others)]:
                if value is None:
                    missing[kind] += 1
            if pick is not None and pick not in (1, 2):
                invalid["PRED"] += 1
            if confidence is not None and confidence not in (1, 2, 3, 4, 5):
                invalid["CONF"] += 1
            if others is not None and not (isinstance(others, (int, float)) and 0 <= others <= 100):
                invalid["OTHERS"] += 1
            if not (pick in (1, 2) and confidence in (1, 2, 3, 4, 5) and
                    isinstance(others, (int, float)) and 0 <= others <= 100):
                continue
            n_complete += 1
            valid_active.add(pid)
            records.append({"participant_id": pid, "source_row": row_num, "game_id": game_id,
                            "study_wave": game_id.split("_")[0], "pick_code": int(pick),
                            "confidence_ordinal": int(confidence), "predicted_peer_agreement_percent": others,
                            "winner_code": int(winner)})
        coverage.append({"game_id": game_id, "study_wave": game_id.split("_")[0],
                         "winner_code": int(winner), "valid_picks": n_picks,
                         "complete_three_answer_records": n_complete})
    assert len(records) == 9788
    assert len({(r["participant_id"], r["game_id"]) for r in records}) == len(records)
    counts = [r["complete_three_answer_records"] for r in coverage]
    assert min(counts) == 10 and max(counts) == 61
    CACHE.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (CACHE / "nba_complete_responses.jsonl").open("w") as f:
        for record in records:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
    with (OUTPUT / "nba_game_coverage.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(coverage[0]))
        writer.writeheader()
        writer.writerows(coverage)
    metadata = {
        "checked_date": "2026-09-24", "dataset": "Rutchick et al. 2020, NBA Study 3",
        "paper": "https://link.springer.com/article/10.1186/s41235-020-00256-z",
        "repository": "https://osf.io/2rzcs/", "download": "https://osf.io/download/zwsgu/",
        "source_sha256": digest, "sheet": sheet, "participant_column": "A", "header_row": 1,
        "participant_rows": len(people), "participants_with_valid_picks": len(raw_active),
        "participants_with_complete_answers": len(valid_active), "games": len(game_ids),
        "valid_pick_records": sum(r["valid_picks"] for r in coverage),
        "complete_three_answer_records": len(records), "complete_responses_per_game_min": min(counts),
        "complete_responses_per_game_max": max(counts),
        "complete_cohort_size_distribution": dict(sorted(Counter(counts).items())),
        "missing_cells": dict(missing), "invalid_nonmissing_cells": dict(invalid),
        "outcome": "The released winner row, code 1 or 2; not bookmaker odds or a model forecast",
        "analysis_status": "Dataset verified and prepared. Panel scoring is a separate workflow: research/sports_survey_equivalence.py. No betting returns calculated.",
        "differences_from_paper": ["Release has 113 active complete-answer participants; published text reports 111 making predictions.",
                                   "Downloaded Appendix A gives slightly different aggregate accuracy figures from the final paper. Treat release as a version-specific reanalysis, not exact replication."],
        "betting_join_status": "Companion file has 465 dates/team pairs and model probabilities, but no explicit mapping to survey wave/game codes and no bookmaker odds. Row-order join not assumed.",
        "proposed_primary_panel_range": [1, 10],
        "why_range": "Same 465 games and same eligible person pool per game across all three methods and panel sizes",
        "checks": {"original_hash_matches_osf": True, "unique_participant_ids": True,
                   "unique_game_ids": True, "unique_person_game_pairs": True,
                   "valid_winner_for_every_game": True, "at_least_ten_complete_people_every_game": True},
        "prepared_local_file": str((CACHE / "nba_complete_responses.jsonl").relative_to(ROOT))}
    (OUTPUT / "inspection.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=CACHE / "nba_data.xlsx")
    prepare(parser.parse_args().input)
