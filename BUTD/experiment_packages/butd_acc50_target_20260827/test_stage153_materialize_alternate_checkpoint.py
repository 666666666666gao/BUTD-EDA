#!/usr/bin/env python3
import os
import tempfile
import unittest

import torch

import stage153_build_final_artifact as artifact
import stage153_materialize_alternate_checkpoint as materializer
from test_stage153_build_final_artifact import ArtifactBuilderTest


class MaterializerTest(unittest.TestCase):
    def test_materialized_checkpoint_replaces_only_locked_head(self):
        with tempfile.TemporaryDirectory() as root:
            args = ArtifactBuilderTest().fixture(root)
            artifact.build_artifact(args)
            output = os.path.join(root, "materialized.pth")
            receipt = materializer.materialize(args.output_dir, output)
            self.assertTrue(os.path.isfile(output))
            self.assertTrue(receipt["temporary_and_removable_after_fresh_inference"])
            primary = torch.load(args.primary_checkpoint, map_location="cpu")
            alternate = torch.load(args.alternate_checkpoint, map_location="cpu")
            materialized = torch.load(output, map_location="cpu")
            self.assertTrue(torch.equal(
                materialized["model"]["shared.weight"],
                primary["model"]["shared.weight"],
            ))
            for name in artifact.ALTERNATE_HEAD_KEYS:
                self.assertTrue(torch.equal(
                    materialized["model"][name], alternate["model"][name]
                ))

    def test_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            args = ArtifactBuilderTest().fixture(root)
            artifact.build_artifact(args)
            output = os.path.join(root, "exists.pth")
            with open(output, "wb") as handle:
                handle.write(b"keep")
            with self.assertRaises(AssertionError):
                materializer.materialize(args.output_dir, output)
            with open(output, "rb") as handle:
                self.assertEqual(handle.read(), b"keep")


if __name__ == "__main__":
    unittest.main()
