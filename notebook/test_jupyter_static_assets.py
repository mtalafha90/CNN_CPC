"""Exercise the Jupyter browser-asset path missed by kernel-only smoke tests.

JupyterLab 4.6.3 / Jupyter Server 2.21.0 with Tornado 6.5.9 raises
AttributeError('allowed_symlink_directory') here. This uses real installed
assets and the real handler without needing a listening socket or a GPU.
"""
from pathlib import Path
from unittest.mock import Mock

import jupyter_server
import jupyterlab
from jupyter_server.base.handlers import FileFindHandler
import pytest
from tornado.httputil import HTTPServerRequest
from tornado.web import Application, HTTPError


@pytest.mark.parametrize("asset_type", ["javascript", "favicon"])
def test_jupyter_resolves_and_validates_shipped_browser_asset(asset_type, monkeypatch):
    if asset_type == "javascript":
        root = Path(jupyterlab.__file__).parent / "static"
        assets = sorted(root.glob("main.*.js"))
        assert assets, "The installed JupyterLab frontend bundle is missing."
        asset = assets[0]
    else:
        root = Path(jupyter_server.__file__).parent / "static/favicons"
        asset = root / "favicon.ico"
    assert asset.is_file() and asset.stat().st_size > 0
    monkeypatch.setattr(FileFindHandler, "_static_paths", {})
    request = HTTPServerRequest(method="GET", uri="/static/" + asset.name, connection=Mock())
    handler = FileFindHandler(Application(), request, path=[str(root)])
    handler.path = handler.parse_url_path(asset.name)
    absolute = handler.get_absolute_path(handler.root, asset.name)
    assert handler.validate_absolute_path(handler.root, absolute) == str(asset.resolve())


def test_static_validation_still_rejects_an_absolute_path_outside_the_root(tmp_path):
    root = tmp_path / "static"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside the allowed static directory")
    request = HTTPServerRequest(method="GET", uri="/static/outside.txt", connection=Mock())
    handler = FileFindHandler(Application(), request, path=[str(root)])
    handler.path = handler.parse_url_path("../outside.txt")
    with pytest.raises(HTTPError) as exc:
        handler.validate_absolute_path(handler.root, str(outside))
    assert exc.value.status_code == 403
