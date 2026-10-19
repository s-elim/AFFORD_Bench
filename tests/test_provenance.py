"""Provenance records: git state parsing."""
from afford_pro import provenance


def test_modified_files_keep_their_full_path_when_the_first_line_starts_with_a_space(monkeypatch):
    status = " M afford_pro/datasets/generation.py\n M scripts/train_policy.py\nM  README.md\n"
    outputs = {("status", "--porcelain", "--untracked-files=no"): status,
               ("rev-parse", "HEAD"): "abc\n", ("rev-parse", "--abbrev-ref", "HEAD"): "main\n"}
    monkeypatch.setattr(provenance, "_git_raw", lambda *args: outputs[args])

    state = provenance.git_state()

    assert state["modified_tracked_files"] == ["README.md", "afford_pro/datasets/generation.py",
                                               "scripts/train_policy.py"]
    assert state["commit"] == "abc" and state["branch"] == "main" and state["dirty"] is True


def test_clean_tree_is_not_dirty(monkeypatch):
    outputs = {("status", "--porcelain", "--untracked-files=no"): "", ("rev-parse", "HEAD"): "abc\n",
               ("rev-parse", "--abbrev-ref", "HEAD"): "main\n"}
    monkeypatch.setattr(provenance, "_git_raw", lambda *args: outputs[args])

    assert provenance.git_state()["dirty"] is False


def test_missing_git_gives_unknown_state(monkeypatch):
    monkeypatch.setattr(provenance, "_git_raw", lambda *args: None)

    state = provenance.git_state()

    assert state == {"commit": None, "branch": None, "dirty": None, "modified_tracked_files": []}
