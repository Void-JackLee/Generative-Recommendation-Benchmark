"""Convert test.csv and id2name4Rec.json into baseline-compatible pickles."""

import argparse
import ast
import json
from pathlib import Path

import pandas as pd


def load_sequences(path):
    frame = pd.read_csv(path, usecols=["user_id", "item_ids", "timestamp"])
    if frame.empty or frame["user_id"].isna().any():
        raise ValueError(f"{path}: empty data or missing user IDs")
    if not pd.api.types.is_integer_dtype(frame["user_id"]):
        raise ValueError(f"{path}: user IDs must be integers")
    if frame["user_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate user IDs")
    for column in ["item_ids", "timestamp"]:
        frame[column] = frame[column].map(ast.literal_eval)
        for user_id, values in zip(frame["user_id"], frame[column]):
            if not isinstance(values, list) or not all(type(v) is int for v in values):
                raise ValueError(f"{path}: user {user_id}, {column} must be a list of integers")
    for row in frame.itertuples(index=False):
        if len(row.item_ids) != len(row.timestamp):
            raise ValueError(f"{path}: user {row.user_id}, item/timestamp length mismatch")
        if any(a > b for a, b in zip(row.timestamp, row.timestamp[1:])):
            raise ValueError(f"{path}: user {row.user_id}, timestamps are not ascending")
    return frame.set_index("user_id")


def convert(source_dir, output_dir, suffix=""):
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    destinations = [output_dir / "user2item.pkl", output_dir / "item2title.pkl"]
    for path in destinations:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")

    sequence_file = f"test_{suffix}.csv" if suffix else "test.csv"
    full = load_sequences(source_dir / sequence_file)
    if full["item_ids"].map(len).min() < 4:
        raise ValueError("Each full sequence needs at least 4 items to produce a training pair")

    titles = json.loads((source_dir / "id2name4Rec.json").read_text(encoding="utf-8"))
    records = []
    for raw_id, title in titles.items():
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"Item {raw_id} must have a non-empty string title")
        records.append({
            "ItemID": int(raw_id),
            "Title": title,
        })
    items = pd.DataFrame(records).sort_values("ItemID").reset_index(drop=True)
    if items["ItemID"].duplicated().any():
        raise ValueError("Title mapping contains duplicate integer item IDs")
    interaction_ids = {item for sequence in full["item_ids"] for item in sequence}
    missing = interaction_ids - set(items["ItemID"])
    if missing:
        raise ValueError(f"Titles missing for {len(missing)} items: {sorted(missing)[:10]}")

    interactions = full.reset_index().rename(columns={
        "user_id": "UserID", "item_ids": "ItemID", "timestamp": "Timestamp",
    })[["UserID", "ItemID", "Timestamp"]]
    output_dir.mkdir(parents=True, exist_ok=True)
    for frame, path in zip([interactions, items], destinations):
        with path.open("xb") as stream:
            frame.to_pickle(stream, protocol=4)
        pd.testing.assert_frame_equal(frame, pd.read_pickle(path))

    lengths = interactions["ItemID"].map(len)
    print(json.dumps({
        "source_dir": str(source_dir),
        "output_files": [str(path) for path in destinations],
        "users": len(interactions),
        "title_items": len(items),
        "interacted_items": len(interaction_ids),
        "interactions": int(lengths.sum()),
        "sequence_length_min": int(lengths.min()),
        "sequence_length_max": int(lengths.max()),
        "original_ids_and_order_preserved": True,
        "pickle_round_trip_verified": True,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--suffix", type=int, default=None)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    convert(args.source_dir, args.output_dir or args.source_dir / (f"pkl_{args.suffix}" if args.suffix else "pkl"), suffix=args.suffix)
