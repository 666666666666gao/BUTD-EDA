import math

import pytest

from main_utils import ValidationEarlyStopper


METRIC = "last__bbs_acc0.25_top1"


def make_stopper():
    return ValidationEarlyStopper(
        metric=METRIC,
        min_epoch=35,
        patience=4,
        min_delta=0.001,
        max_epoch=100,
        val_freq=5,
    )


def update(stopper, epoch, score):
    return stopper.update(epoch, {METRIC: score})


def test_stops_only_after_four_stale_validations_at_or_after_min_epoch():
    stopper = make_stopper()
    for epoch, score in (
        (5, 0.35), (10, 0.41), (15, 0.46),
        (20, 0.48), (25, 0.483), (30, 0.4946),
    ):
        assert not update(stopper, epoch, score)["should_stop"]
    for epoch, score in ((35, 0.4947), (40, 0.4945), (45, 0.4950)):
        event = update(stopper, epoch, score)
        assert not event["should_stop"]
    event = update(stopper, 50, 0.4949)
    assert event["should_stop"]
    assert event["stale_validations"] == 4
    assert event["reference_epoch"] == 30


def test_meaningful_gain_resets_patience_but_tiny_gain_does_not():
    stopper = make_stopper()
    update(stopper, 30, 0.5000)
    update(stopper, 35, 0.5005)
    assert stopper.stale_validations == 1
    event = update(stopper, 40, 0.5011)
    assert event["meaningful_improvement"]
    assert event["stale_validations"] == 0
    assert event["reference_epoch"] == 40


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
def test_rejects_non_finite_metric(score):
    with pytest.raises(ValueError):
        update(make_stopper(), 5, score)


def test_receipt_records_policy_and_terminal_reason():
    stopper = make_stopper()
    update(stopper, 5, 0.4)
    receipt = stopper.receipt("early_stopped", stop_epoch=50)
    assert receipt["metric"] == METRIC
    assert receipt["min_epoch"] == 35
    assert receipt["patience_validations"] == 4
    assert receipt["min_delta"] == 0.001
    assert receipt["validation_frequency_epochs"] == 5
    assert receipt["maximum_epoch"] == 100
    assert receipt["stop_epoch"] == 50
    assert receipt["reason"] == "validation_metric_saturated"
