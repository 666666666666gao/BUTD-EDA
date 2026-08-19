"""Train/evaluate a frozen source-choice calibrator from dumped features."""

import argparse
from collections import OrderedDict, namedtuple
import json
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier


DEFAULT_SOURCE_NAMES = (
    "base",
    "fused",
    "quality",
    "contrastive_base",
    "acd",
)
CandidateGroup = namedtuple(
    "CandidateGroup", ["example_id", "indices", "target_offset"]
)
LABEL_KEYS = {
    "oracle_source_id",
    "threshold_utility_source_id",
    "oracle_iou",
    "oracle_hit025",
    "oracle_hit050",
}
CANDIDATE_LABEL_KEYS = {
    "example_id",
    "candidate_query",
    "candidate_iou",
    "candidate_hit025",
    "candidate_hit050",
    "threshold_utility",
    "oracle_candidate",
}
METADATA_KEYS = {
    "source_names",
    "row_type",
    "source_row_type",
}


def _is_feature_key(key):
    if key in LABEL_KEYS:
        return False
    if key in METADATA_KEYS:
        return False
    if key == "example_id":
        return False
    if key.endswith("_source_id"):
        return False
    if key.endswith("_top_iou"):
        return False
    return True


def _is_candidate_feature_key(key):
    return key not in CANDIDATE_LABEL_KEYS and key not in METADATA_KEYS


def _normalize_source_names(source_names=None):
    if source_names is None:
        return DEFAULT_SOURCE_NAMES
    if isinstance(source_names, str):
        source_names = [
            item.strip()
            for item in source_names.split(",")
            if item.strip()
        ]
    return tuple(str(source) for source in source_names)


def _load_rows_with_metadata(path):
    data = torch.load(path, map_location="cpu")
    metadata = (
        {key: value for key, value in data.items() if key != "rows"}
        if isinstance(data, dict)
        else {}
    )
    if isinstance(data, dict) and data.get("format") == (
        "source_choice_feature_dump_sharded_v1"
    ):
        base_dir = os.path.dirname(path)
        rows = []
        for shard in data.get("shards", []):
            shard_path = shard
            if not os.path.isabs(shard_path):
                shard_path = os.path.join(base_dir, shard_path)
            shard_data = torch.load(shard_path, map_location="cpu")
            shard_rows = (
                shard_data["rows"]
                if isinstance(shard_data, dict)
                else shard_data
            )
            if not isinstance(shard_rows, list):
                raise ValueError(f"{shard_path} does not contain a row list")
            rows.extend(shard_rows)
        expected_count = data.get("row_count")
        if expected_count is not None and int(expected_count) != len(rows):
            raise ValueError(
                "row count mismatch for {}: expected {}, got {}".format(
                    path, expected_count, len(rows)
                )
            )
        return rows, metadata
    rows = data["rows"] if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not contain a row list")
    return rows, metadata


def _load_dump_metadata(path):
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        return {}
    return {key: value for key, value in data.items() if key != "rows"}


def _resolve_source_names(source_names=None, metadata=None):
    if source_names is not None:
        return _normalize_source_names(source_names)
    metadata = metadata or {}
    for key in ("source_names", "selector_choice_source_names"):
        if key in metadata and metadata[key]:
            return _normalize_source_names(metadata[key])
    return DEFAULT_SOURCE_NAMES


def build_feature_matrix(rows, label_key="oracle_source_id", columns=None):
    if not rows:
        raise ValueError("feature rows are empty")
    if columns is None:
        columns = sorted(k for k in rows[0].keys() if _is_feature_key(k))
    x = np.asarray(
        [[float(row.get(column, 0.0)) for column in columns] for row in rows],
        dtype=np.float32,
    )
    y = np.asarray([int(row[label_key]) for row in rows], dtype=np.int64)
    return x, y, columns


def build_candidate_feature_matrix(
    rows, label_key="oracle_candidate", columns=None
):
    if not rows:
        raise ValueError("candidate rows are empty")
    if columns is None:
        columns = sorted(
            k for k in rows[0].keys() if _is_candidate_feature_key(k)
        )
    x = np.asarray(
        [[float(row.get(column, 0.0)) for column in columns] for row in rows],
        dtype=np.float32,
    )
    y = np.asarray([int(row[label_key]) for row in rows], dtype=np.int64)
    return x, y, columns


def build_candidate_groups(rows):
    if not rows:
        raise ValueError("candidate rows are empty")

    groups = OrderedDict()
    for row_idx, row in enumerate(rows):
        example_id = row.get("example_id", float(row_idx))
        if example_id not in groups:
            groups[example_id] = {"indices": [], "target_offset": None}
        group = groups[example_id]
        group["indices"].append(row_idx)
        if float(row.get("oracle_candidate", 0.0)) > 0.5:
            target_offset = len(group["indices"]) - 1
            if group["target_offset"] is None:
                group["target_offset"] = target_offset

    result = []
    for example_id, group in groups.items():
        if group["target_offset"] is None:
            raise ValueError(
                "candidate group {} has no oracle candidate".format(
                    example_id
                )
            )
        result.append(
            CandidateGroup(
                example_id=example_id,
                indices=group["indices"],
                target_offset=group["target_offset"],
            )
        )
    return result


class CandidateListwiseScorer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, feature_mean, feature_std):
        super().__init__()
        self.register_buffer(
            "feature_mean",
            torch.as_tensor(feature_mean, dtype=torch.float32),
        )
        self.register_buffer(
            "feature_std",
            torch.as_tensor(feature_std, dtype=torch.float32),
        )
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        x = (x - self.feature_mean) / self.feature_std
        return self.net(x).squeeze(-1)


def evaluate_source_predictions(rows, predictions):
    return evaluate_source_predictions_with_source_names(
        rows, predictions, source_names=None
    )


def evaluate_source_predictions_with_source_names(
    rows, predictions, source_names=None
):
    predictions = list(predictions)
    if len(rows) != len(predictions):
        raise ValueError("rows and predictions must have the same length")
    source_names = _resolve_source_names(source_names)
    ious = []
    for row, source_idx in zip(rows, predictions):
        source = source_names[int(source_idx)]
        ious.append(float(row.get(f"{source}_top_iou", -1.0)))
    ious = np.asarray(ious, dtype=np.float32)
    return {
        "count": int(len(rows)),
        "mean_iou": float(ious.mean()) if len(ious) else 0.0,
        "acc025": float((ious > 0.25).mean()) if len(ious) else 0.0,
        "acc050": float((ious > 0.50).mean()) if len(ious) else 0.0,
    }


def evaluate_candidate_scores(rows, scores):
    scores = list(scores)
    if len(rows) != len(scores):
        raise ValueError("rows and scores must have the same length")

    best_by_example = {}
    for idx, (row, score) in enumerate(zip(rows, scores)):
        example_id = row.get("example_id", idx)
        item = best_by_example.get(example_id, None)
        if item is None or float(score) > item[0]:
            best_by_example[example_id] = (float(score), row)

    ious = np.asarray(
        [
            float(row.get("candidate_iou", -1.0))
            for _, row in best_by_example.values()
        ],
        dtype=np.float32,
    )
    return {
        "count": int(len(ious)),
        "mean_iou": float(ious.mean()) if len(ious) else 0.0,
        "acc025": float((ious > 0.25).mean()) if len(ious) else 0.0,
        "acc050": float((ious > 0.50).mean()) if len(ious) else 0.0,
    }


def _resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _listwise_epoch(model, x_train, groups, optimizer, batch_size, device):
    order = np.random.permutation(len(groups))
    total_loss = 0.0
    total_groups = 0
    model.train()
    for start in range(0, len(order), batch_size):
        batch_groups = [groups[int(idx)] for idx in order[start:start + batch_size]]
        row_indices = []
        lengths = []
        targets = []
        for group in batch_groups:
            row_indices.extend(group.indices)
            lengths.append(len(group.indices))
            targets.append(group.target_offset)

        row_indices = torch.as_tensor(row_indices, dtype=torch.long, device=device)
        raw_scores = model(x_train.index_select(0, row_indices))
        max_len = max(lengths)
        scores = raw_scores.new_full((len(batch_groups), max_len), -1.0e9)
        offset = 0
        for batch_idx, length in enumerate(lengths):
            scores[batch_idx, :length] = raw_scores[offset:offset + length]
            offset += length
        targets = torch.as_tensor(targets, dtype=torch.long, device=device)
        loss = F.cross_entropy(scores, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu().item()) * len(batch_groups)
        total_groups += len(batch_groups)
    return total_loss / max(total_groups, 1)


def _predict_candidate_listwise_scores(model, rows, columns, device):
    x, _, _ = build_candidate_feature_matrix(rows, columns=columns)
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, x.shape[0], 8192):
            batch_scores = model(x[start:start + 8192])
            scores.extend(batch_scores.detach().cpu().numpy().tolist())
    return scores


def _load_rows(path):
    rows, _ = _load_rows_with_metadata(path)
    return rows


def _source_counts(values, source_names=None):
    source_names = _resolve_source_names(source_names)
    counts = {source: 0 for source in source_names}
    for value in values:
        counts[source_names[int(value)]] += 1
    return counts


def train_calibrator(
    train_rows, val_rows, label_key="oracle_source_id", source_names=None,
    metadata=None,
):
    source_names = _resolve_source_names(source_names, metadata=metadata)
    x_train, y_train, columns = build_feature_matrix(
        train_rows, label_key=label_key
    )
    x_val, y_val, _ = build_feature_matrix(
        val_rows, label_key=label_key, columns=columns
    )
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=8,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=0,
    )
    model.fit(x_train, y_train)
    train_pred = model.predict(x_train)
    val_pred = model.predict(x_val)
    metrics = {
        "label_key": label_key,
        "source_names": list(source_names),
        "num_features": len(columns),
        "feature_columns": columns,
        "train_label_counts": _source_counts(y_train, source_names),
        "val_label_counts": _source_counts(y_val, source_names),
        "train_pred_counts": _source_counts(train_pred, source_names),
        "val_pred_counts": _source_counts(val_pred, source_names),
        "train": evaluate_source_predictions_with_source_names(
            train_rows, train_pred, source_names=source_names
        ),
        "val": evaluate_source_predictions_with_source_names(
            val_rows, val_pred, source_names=source_names
        ),
        "train_oracle": evaluate_source_predictions_with_source_names(
            train_rows, y_train, source_names=source_names
        ),
        "val_oracle": evaluate_source_predictions_with_source_names(
            val_rows, y_val, source_names=source_names
        ),
    }
    return model, metrics


def train_candidate_calibrator(train_rows, val_rows):
    x_train, y_train, columns = build_candidate_feature_matrix(train_rows)
    x_val, y_val, _ = build_candidate_feature_matrix(
        val_rows, columns=columns
    )
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=8,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=0,
    )
    model.fit(x_train, y_train)
    train_scores = model.predict_proba(x_train)[:, 1]
    val_scores = model.predict_proba(x_val)[:, 1]
    metrics = {
        "row_type": "candidate",
        "num_features": len(columns),
        "feature_columns": columns,
        "train_positive_rows": int(y_train.sum()),
        "val_positive_rows": int(y_val.sum()),
        "train": evaluate_candidate_scores(train_rows, train_scores),
        "val": evaluate_candidate_scores(val_rows, val_scores),
        "train_oracle": evaluate_candidate_scores(train_rows, y_train),
        "val_oracle": evaluate_candidate_scores(val_rows, y_val),
    }
    return model, metrics


def train_candidate_listwise_calibrator(
    train_rows,
    val_rows,
    hidden_dim=64,
    epochs=20,
    lr=0.001,
    batch_size=128,
    weight_decay=0.0001,
    seed=0,
    device="auto",
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = _resolve_device(device)

    x_train, y_train, columns = build_candidate_feature_matrix(train_rows)
    build_candidate_feature_matrix(val_rows, columns=columns)
    train_groups = build_candidate_groups(train_rows)
    val_groups = build_candidate_groups(val_rows)

    feature_mean = x_train.mean(axis=0)
    feature_std = x_train.std(axis=0)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
    model = CandidateListwiseScorer(
        input_dim=x_train.shape[1],
        hidden_dim=hidden_dim,
        feature_mean=feature_mean,
        feature_std=feature_std,
    ).to(device)
    x_train_tensor = torch.as_tensor(
        x_train, dtype=torch.float32, device=device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    losses = []
    for _ in range(epochs):
        losses.append(
            _listwise_epoch(
                model, x_train_tensor, train_groups, optimizer,
                batch_size=batch_size, device=device,
            )
        )

    train_scores = _predict_candidate_listwise_scores(
        model, train_rows, columns, device
    )
    val_scores = _predict_candidate_listwise_scores(
        model, val_rows, columns, device
    )
    metrics = {
        "row_type": "candidate_listwise",
        "num_features": len(columns),
        "feature_columns": columns,
        "hidden_dim": int(hidden_dim),
        "epochs": int(epochs),
        "lr": float(lr),
        "batch_size": int(batch_size),
        "weight_decay": float(weight_decay),
        "device": device,
        "final_loss": float(losses[-1]) if losses else 0.0,
        "train_positive_rows": int(y_train.sum()),
        "val_positive_rows": int(
            sum(float(row.get("oracle_candidate", 0.0)) > 0.5 for row in val_rows)
        ),
        "train_groups": int(len(train_groups)),
        "val_groups": int(len(val_groups)),
        "train": evaluate_candidate_scores(train_rows, train_scores),
        "val": evaluate_candidate_scores(val_rows, val_scores),
        "train_oracle": evaluate_candidate_scores(train_rows, y_train),
        "val_oracle": evaluate_candidate_scores(
            val_rows,
            [
                float(row.get("oracle_candidate", 0.0))
                for row in val_rows
            ],
        ),
    }
    model = model.to("cpu")
    return model, metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--source_names",
        default="",
        help=(
            "Comma-separated source-name order for source rows. "
            "Defaults to metadata in the dumps or the legacy source list."
        ),
    )
    parser.add_argument(
        "--label_key",
        default="oracle_source_id",
        choices=["oracle_source_id", "threshold_utility_source_id"],
    )
    parser.add_argument(
        "--row_type",
        default="source",
        choices=["source", "candidate", "candidate_listwise"],
    )
    parser.add_argument("--listwise_hidden_dim", type=int, default=64)
    parser.add_argument("--listwise_epochs", type=int, default=20)
    parser.add_argument("--listwise_lr", type=float, default=0.001)
    parser.add_argument("--listwise_batch_size", type=int, default=128)
    parser.add_argument("--listwise_weight_decay", type=float, default=0.0001)
    parser.add_argument("--listwise_seed", type=int, default=0)
    parser.add_argument("--listwise_device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    train_rows, train_metadata = _load_rows_with_metadata(args.train)
    val_rows, val_metadata = _load_rows_with_metadata(args.val)
    source_names = _resolve_source_names(
        args.source_names or None,
        metadata=train_metadata if train_metadata else val_metadata,
    )
    if args.row_type == "candidate_listwise":
        model, metrics = train_candidate_listwise_calibrator(
            train_rows,
            val_rows,
            hidden_dim=args.listwise_hidden_dim,
            epochs=args.listwise_epochs,
            lr=args.listwise_lr,
            batch_size=args.listwise_batch_size,
            weight_decay=args.listwise_weight_decay,
            seed=args.listwise_seed,
            device=args.listwise_device,
        )
    elif args.row_type == "candidate":
        model, metrics = train_candidate_calibrator(train_rows, val_rows)
    else:
        model, metrics = train_calibrator(
            train_rows,
            val_rows,
            label_key=args.label_key,
            source_names=source_names,
            metadata=train_metadata if train_metadata else val_metadata,
        )
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
