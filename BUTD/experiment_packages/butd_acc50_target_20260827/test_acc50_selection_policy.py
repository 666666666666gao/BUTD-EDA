#!/usr/bin/env python
"""Self-test the preservation-first checkpoint policy for the Acc@0.50 goal."""

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from main_utils import (
    _best_checkpoint_constraint_snapshot,
    _constraint_selection_promotes,
)


METRIC_KEYS = (
    'last__bbs_unique_acc0.25_top1',
    'last__bbs_unique_acc0.50_top1',
    'last__bbs_multiple_acc0.25_top1',
    'last__bbs_multiple_acc0.50_top1',
    'last__bbs_acc0.25_top1',
    'last__bbs_acc0.50_top1',
)
LOWER = [0.0, 0.0, 0.0, 0.0, 0.5391, 0.4241]
N_VAL = 9508


def results(overall_025, overall_050):
    values = [0.85, 0.65, 0.48, 0.35, overall_025, overall_050]
    return dict(zip(METRIC_KEYS, values))


def selection(args, overall_025, overall_050):
    snapshot = _best_checkpoint_constraint_snapshot(
        args, results(overall_025, overall_050)
    )
    snapshot['score'] = overall_025
    return snapshot


def main():
    args = SimpleNamespace(
        best_checkpoint_constraint_lower=LOWER,
        best_checkpoint_constraint_upper=None,
        best_checkpoint_constraint_epsilon=0.0,
    )

    equal_025 = selection(args, 0.5391, 0.43)
    equal_050 = selection(args, 0.54, 0.4241)
    assert not equal_025['constraint_feasible']
    assert not equal_050['constraint_feasible']

    min_hits_025 = 5126
    min_hits_050 = 4033
    min_grid_025 = min_hits_025 / N_VAL
    min_grid_050 = min_hits_050 / N_VAL
    assert (min_hits_025 - 1) / N_VAL <= 0.5391 < min_grid_025
    assert (min_hits_050 - 1) / N_VAL <= 0.4241 < min_grid_050
    grid_feasible = selection(args, min_grid_025, min_grid_050)
    assert grid_feasible['constraint_feasible']

    preserve = selection(args, 0.5440, 0.4250)
    higher_050_but_lower_025 = selection(args, 0.5410, 0.4500)
    assert preserve['constraint_feasible']
    assert higher_050_but_lower_025['constraint_feasible']
    assert not _constraint_selection_promotes(
        preserve,
        higher_050_but_lower_025,
        higher_050_but_lower_025['score'],
        min_delta=0.0,
    )

    higher_025_still_feasible = selection(args, 0.5450, 0.4242)
    assert higher_025_still_feasible['constraint_feasible']
    assert _constraint_selection_promotes(
        preserve,
        higher_025_still_feasible,
        higher_025_still_feasible['score'],
        min_delta=0.0,
    )

    infeasible = selection(args, 0.55, 0.42)
    assert not infeasible['constraint_feasible']
    assert _constraint_selection_promotes(
        infeasible,
        preserve,
        preserve['score'],
        min_delta=0.0,
    )

    far_from_acc050 = selection(args, 0.5440, 0.4100)
    closer_to_both = selection(args, 0.5420, 0.4230)
    assert not far_from_acc050['constraint_feasible']
    assert not closer_to_both['constraint_feasible']
    assert _constraint_selection_promotes(
        far_from_acc050,
        closer_to_both,
        closer_to_both['score'],
        min_delta=0.0,
    )

    sacrifices_acc025 = selection(args, 0.5300, 0.4240)
    assert not sacrifices_acc025['constraint_feasible']
    assert not _constraint_selection_promotes(
        closer_to_both,
        sacrifices_acc025,
        sacrifices_acc025['score'],
        min_delta=0.0,
    )

    print('ACC50_PRESERVATION_POLICY_SELFTEST_PASS')
    print('minimum_grid_acc0.25={:.10f}'.format(min_grid_025))
    print('minimum_grid_acc0.50={:.10f}'.format(min_grid_050))


if __name__ == '__main__':
    main()
