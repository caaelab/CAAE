from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_configs_do_not_expose_ablation_controls():
    forbidden_training_keys = {"rec", "con", "adv"}
    forbidden_evaluation_keys = {
        "smoothing",
        "score_postprocess",
        "threshold_refine",
        "fixed_alpha",
    }

    for config_path in sorted((ROOT / "configs").glob("config_*.yaml")):
        config = load_yaml(config_path)
        training_keys = set(config.get("training", {}))
        evaluation_keys = set(config.get("evaluation", {}))

        assert forbidden_training_keys.isdisjoint(training_keys), config_path.name
        assert forbidden_evaluation_keys.isdisjoint(evaluation_keys), config_path.name


def test_source_does_not_keep_alternate_or_disable_paths():
    train_source = (ROOT / "train.py").read_text(encoding="utf-8")
    eval_source = (ROOT / "eval.py").read_text(encoding="utf-8")
    loss_source = (ROOT / "model" / "loss.py").read_text(encoding="utf-8")

    assert "config['training'].get('rec'" not in train_source
    assert "config['training'].get('con'" not in train_source
    assert "config['training'].get('adv'" not in train_source
    assert "class InfoNCELoss" not in loss_source
    assert "fixed_alpha" not in eval_source
    assert '"none"' not in eval_source
    assert '"off"' not in eval_source
    assert '"false"' not in eval_source
