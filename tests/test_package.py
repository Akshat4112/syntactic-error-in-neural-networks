import os
import subprocess
import sys

from syntaxbench.dataset import dataset


def test_workspace_defaults_to_current_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert dataset().project_root == tmp_path.resolve()


def test_explicit_workspace(tmp_path):
    assert dataset(project_root=tmp_path).project_root == tmp_path.resolve()


def test_cli_help_outside_checkout(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "syntaxbench", "--help"],
        cwd=tmp_path, capture_output=True, text=True,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        check=True,
    )
    assert "--project-root" in result.stdout
    assert "preprocess" in result.stdout


def test_package_cli_import_is_lightweight(tmp_path):
    subprocess.run(
        [sys.executable, "-c",
         "import sys; import syntaxbench.cli; "
         "assert 'tensorflow' not in sys.modules; "
         "assert 'torch' not in sys.modules"],
        cwd=tmp_path, check=True,
    )


def test_active_imports_preserve_explicit_environment(tmp_path):
    env = dict(os.environ, WANDB_DISABLED="sentinel")
    subprocess.run(
        [sys.executable, "-c",
         "import os; import syntaxbench.dataset, syntaxbench.train, "
         "syntaxbench.evaluation, syntaxbench.transformer; "
         "assert os.environ['WANDB_DISABLED'] == 'sentinel'"],
        cwd=tmp_path, env=env, check=True,
    )
    assert not list(tmp_path.iterdir())
