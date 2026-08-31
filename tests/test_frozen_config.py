"""
Formalizes the ad hoc scratchpad verification of the frozen-configuration
system (src/training/reporting.py): feature-audit dimensions, final-run-
configuration fields (including git commit hash / software versions),
and the write/guard round-trip (raises on a mismatched re-freeze, no-op on
an identical one).

Requires a locally cached (or downloadable) facebook/wav2vec2-base-960h
checkpoint, same as tests/test_gated_fusion_shapes.py.

Run with: pytest tests/test_frozen_config.py -v
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.training.reporting import (build_final_run_configuration, check_frozen_config_guard,
                                    feature_audit, print_final_run_configuration,
                                    write_frozen_config)
from src.training.runner import TrainingConfig


def _synthetic_manifest() -> pd.DataFrame:
    rows = []
    for speaker in config.ALL_SPEAKERS:
        severity = config.SEVERITY_MAP.get(speaker, "N/A (Control)")
        rows.append({"Speaker_ID": speaker, "Severity": severity})
    return pd.DataFrame(rows)


def test_feature_audit_dimensions_match_config():
    audit = feature_audit(num_classes=4)
    assert audit["learned_branch"]["dimensions"] == config.LEARNED_EMBED_DIM == 128
    assert audit["segmental_branch"]["dimensions"] == config.SEGMENTAL_EMBED_DIM == 64
    assert audit["suprasegmental_branch"]["dimensions"] == config.SUPRA_EMBED_DIM == 64
    assert audit["fusion"]["fused_dim"] == config.FUSED_EMBED_DIM == 256
    assert audit["segmental_branch"]["input_channels"] == config.SEGMENTAL_CHANNELS == 43
    assert audit["suprasegmental_branch"]["input_channels"] == config.SUPRA_CHANNELS == 3


def test_final_run_configuration_has_provenance_fields():
    df = _synthetic_manifest()
    cfg = TrainingConfig(task="severity", model="gated_fusion_three_branch",
                         run_name="severity_gated_fusion_three_branch_TEST")
    final_config = build_final_run_configuration(cfg, df, num_speakers_total=28)

    assert "provenance" in final_config
    prov = final_config["provenance"]
    assert "git_commit" in prov and "software_versions" in prov
    # git_commit may be None outside a repo, but must never raise or be a
    # placeholder string — either a real hash or None, nothing else.
    assert prov["git_commit"] is None or isinstance(prov["git_commit"], str)
    versions = prov["software_versions"]
    assert "python" in versions and versions["python"] is not None
    assert "torch" in versions and versions["torch"] is not None   # torch is a hard project dependency


def test_final_run_configuration_matches_frozen_severity_protocol():
    df = _synthetic_manifest()
    cfg = TrainingConfig(task="severity", model="gated_fusion_three_branch",
                         run_name="severity_gated_fusion_three_branch_TEST",
                         severity_protocol="full_loso")
    final_config = build_final_run_configuration(cfg, df, num_speakers_total=28)
    assert final_config["dataset"]["num_folds"] == 15
    assert sorted(final_config["dataset"]["fold_speakers"]) == sorted(config.DYSARTHRIC_IDS)
    assert final_config["complementarity"]["lambda"] == config.LAMBDA_COMP == 0.05
    assert final_config["speaker_invariance"]["lambda"] == config.LAMBDA_SPEAKER == 0.1


def test_print_final_run_configuration_runs_without_raising():
    df = _synthetic_manifest()
    cfg = TrainingConfig(task="severity", model="gated_fusion_three_branch",
                         run_name="severity_gated_fusion_three_branch_TEST")
    final_config = print_final_run_configuration(cfg, df)
    assert final_config["run_name"] == "severity_gated_fusion_three_branch_TEST"


def test_write_and_guard_round_trip():
    df = _synthetic_manifest()
    cfg = TrainingConfig(task="severity", model="gated_fusion_three_branch",
                         run_name="severity_gated_fusion_three_branch_TEST")
    final_config = build_final_run_configuration(cfg, df, num_speakers_total=28)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "frozen_config_test.json"
        write_frozen_config(final_config, path=path)
        assert path.exists()

        # Identical config -> no raise (a legitimate resume).
        check_frozen_config_guard(final_config, path=path)

        # Different run_name -> no-op (the guard only checks matching run_names).
        other_run = dict(final_config)
        other_run["run_name"] = "some_other_run"
        check_frozen_config_guard(other_run, path=path)

        # Same run_name, different hyperparameter -> must raise.
        mutated = dict(final_config)
        mutated["optimizer"] = dict(final_config["optimizer"])
        mutated["optimizer"]["lr_head"] = 999.0
        raised = False
        try:
            check_frozen_config_guard(mutated, path=path)
        except RuntimeError:
            raised = True
        assert raised, "guard did not raise on a same-run-name, different-config mismatch"


def test_guard_is_a_noop_when_no_frozen_config_exists_yet():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "does_not_exist.json"
        # Must not raise — the normal state before the one real freeze.
        check_frozen_config_guard({"run_name": "anything"}, path=path)


if __name__ == "__main__":
    test_feature_audit_dimensions_match_config()
    test_final_run_configuration_has_provenance_fields()
    test_final_run_configuration_matches_frozen_severity_protocol()
    test_print_final_run_configuration_runs_without_raising()
    test_write_and_guard_round_trip()
    test_guard_is_a_noop_when_no_frozen_config_exists_yet()
    print("All frozen-configuration tests passed.")
