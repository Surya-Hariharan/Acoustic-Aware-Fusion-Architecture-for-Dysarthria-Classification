"""
The frozen run configuration (src/training/reporting.py): its fields, the
per-run storage, and the guard that refuses to change a run under its name.
"""

import pytest

from src import config
from src.training import reporting
from src.training.reporting import (build_final_run_configuration, check_frozen_config_guard,
                                    frozen_config_path, write_frozen_config)
from src.training.runner import TrainingConfig


@pytest.fixture
def results_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    return tmp_path


def test_final_configuration_records_protocol_architecture_and_provenance():
    final = build_final_run_configuration(TrainingConfig(run_name="TEST"))
    assert final["dataset"]["num_folds"] == 15
    assert final["dataset"]["fold_speakers"] == config.SEVERITY_LOSO_ORDER
    assert sorted(final["dataset"]["fold_speakers"]) == sorted(config.DYSARTHRIC_IDS)
    assert final["architecture"]["embedding_dims"] == {"learned": 128, "segmental": 64, "supra": 64}
    assert final["architecture"]["lambda_comp"] == config.LAMBDA_COMP
    assert final["optimizer"]["lr_lora"] == config.DEFAULT_LR_LORA
    assert final["ablation_switches"] is None
    prov = final["provenance"]
    assert prov["git_commit"] is None or isinstance(prov["git_commit"], str)
    assert prov["software_versions"]["torch"] is not None


def test_ablation_and_truncation_are_recorded():
    final = build_final_run_configuration(TrainingConfig(model="ab2_acoustic_only",
                                                         run_name="TEST", max_folds=2))
    assert final["ablation_switches"]["branches"] == ["segmental", "supra"]
    assert final["dataset"]["num_folds"] == 2


def test_guard_allows_resume_and_refuses_a_changed_configuration(results_dir):
    final = build_final_run_configuration(TrainingConfig(run_name="TEST"))
    check_frozen_config_guard(final)                     # nothing frozen yet: no-op
    path = write_frozen_config(final)
    assert path == frozen_config_path("TEST") and path.exists()
    check_frozen_config_guard(final)                     # identical config: a resume

    changed = build_final_run_configuration(TrainingConfig(run_name="TEST", lr_head=5e-4))
    with pytest.raises(RuntimeError, match="different configuration"):
        check_frozen_config_guard(changed)


def test_provenance_changes_do_not_trip_the_guard(results_dir, monkeypatch):
    final = build_final_run_configuration(TrainingConfig(run_name="TEST"))
    write_frozen_config(final)
    monkeypatch.setattr(reporting, "_git_commit_hash", lambda: "a-later-commit")
    check_frozen_config_guard(build_final_run_configuration(TrainingConfig(run_name="TEST")))


def test_freezing_an_ablation_never_touches_the_primary_runs_freeze(results_dir):
    primary = build_final_run_configuration(TrainingConfig(run_name="PRIMARY"))
    write_frozen_config(primary)
    write_frozen_config(build_final_run_configuration(
        TrainingConfig(model="ab1_wav2vec2_only", run_name="ABLATION")))

    changed_primary = build_final_run_configuration(TrainingConfig(run_name="PRIMARY", epochs=30))
    with pytest.raises(RuntimeError):
        check_frozen_config_guard(changed_primary)
