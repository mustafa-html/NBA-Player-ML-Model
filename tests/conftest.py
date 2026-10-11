import pytest

from playstyle import config
from playstyle.pipeline import Engine, train


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """Train once on the bundled data (smaller k search to keep tests fast)."""
    root = tmp_path_factory.mktemp("run")
    cfg = config.TrainConfig()
    cfg.k = 10
    cfg.k_range = (9, 10)
    summary = train(cfg, artifact_dir=root / "artifacts", report_dir=root / "reports")
    return {"summary": summary, "artifacts": root / "artifacts", "reports": root / "reports"}


@pytest.fixture(scope="session")
def engine(trained):
    return Engine.load(trained["artifacts"])


@pytest.fixture(scope="session")
def client(trained):
    from webapp import create_app

    app = create_app(trained["artifacts"], trained["reports"])
    app.testing = True
    return app.test_client()
