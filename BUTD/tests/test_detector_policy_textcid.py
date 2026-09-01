import torch

from models.bdetr import BeaUTyDETR


def test_detector_policy_uses_text_cid_during_training_and_eval():
    gt = torch.tensor([4, 5])
    text = torch.tensor([7, 8])
    assert torch.equal(BeaUTyDETR._detector_policy_target_cid({
        'train': True, 'target_cid': gt, 'text_target_cid': text,
    }), text)
    assert torch.equal(BeaUTyDETR._detector_policy_target_cid({
        'train': False, 'target_cid': gt, 'text_target_cid': text,
        'eval_target_cid_source': 'text',
    }), text)
