import tempfile
from types import SimpleNamespace

import torch
import torch.nn as nn

from main_utils import save_best_checkpoint


def main():
    torch.distributed.init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:29591",
        rank=0,
        world_size=1,
    )
    bare = nn.Linear(3, 2)
    ddp = nn.parallel.DistributedDataParallel(bare)
    optimizer = torch.optim.AdamW(ddp.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    with tempfile.TemporaryDirectory() as tmp:
        args = SimpleNamespace(
            log_dir=tmp,
            best_checkpoint_metric="last__bbs_acc0.25_top1",
            best_checkpoint_min_delta=0.0,
        )
        original = {
            key: value.detach().clone() for key, value in ddp.state_dict().items()
        }
        save_best_checkpoint(
            args,
            5,
            ddp,
            optimizer,
            scheduler,
            {"last__bbs_acc0.25_top1": 0.5},
        )
        with torch.no_grad():
            for parameter in ddp.parameters():
                parameter.add_(1.0)
        checkpoint = torch.load(
            tmp + "/ckpt_best_primary.pth", map_location="cpu"
        )
        ddp.load_state_dict(checkpoint["model"], strict=True)
        assert checkpoint["epoch"] == 5
        for key, value in ddp.state_dict().items():
            assert torch.equal(value, original[key])
    torch.distributed.destroy_process_group()
    print("BEST_CHECKPOINT_DDP_RELOAD_PASS")


if __name__ == "__main__":
    main()
