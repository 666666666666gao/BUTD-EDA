# Fixed ScanRefer ablation training protocol: LR55/LR60/E65

Status: deployed; the legacy Full pilot will transition after its complete epoch-55 validation.

All nine formal trained rows use one identical protocol:

- independent start from the verified official detector initialization;
- seed 0, no resume checkpoint;
- exactly 65 training epochs;
- validation every 5 epochs;
- step learning-rate decay by 0.1 after epoch 55;
- a second 0.1 decay after epoch 60;
- no validation-based early termination;
- strict-best official Overall Acc@0.25 checkpoint retained with zero selection delta;
- final evaluation reloads that checkpoint;
- exactly one model-weight file retained for each formal row.

The scheduler implementation was corrected so a no-warm-up milestone is not shifted one epoch late. Unit evidence verifies that milestones 55 and 60 map to 55*N and 60*N optimizer steps.

The pre-freeze Full run used max_epoch=100 and lr_decay_epochs=[65]. It is a tuning pilot and is excluded from the paper table. After its complete epoch-55 validation, the bridge independently reloads/evaluates its strict-best checkpoint, records SHA256, removes that unused pilot weight, archives the pilot logs outside the formal train root, and restarts the Full row from the official initialization under this fixed protocol.
