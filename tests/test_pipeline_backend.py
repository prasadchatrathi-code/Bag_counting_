import pytest

from bagcount.pipeline import _is_checkpoint, _is_onnx, _use_onnx, resolve_onnx_path


def test_checkpoint_extensions(tmp_path):
    for name in ("model.pt", "model.pth", "model.PTH"):
        f = tmp_path / name
        f.write_bytes(b"")
        assert _is_checkpoint(str(f)), name
        assert not _is_onnx(str(f)), name


def test_onnx_file_and_export_dir(tmp_path):
    f = tmp_path / "model.onnx"
    f.write_bytes(b"")
    assert _is_onnx(str(f))
    assert _is_onnx(str(tmp_path))
    assert not _is_checkpoint(str(f))


def test_auto_backend_detection(tmp_path):
    ckpt = tmp_path / "checkpoint.pth"
    ckpt.write_bytes(b"")
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"")
    assert _use_onnx(str(ckpt)) is False
    assert _use_onnx(str(onnx)) is True


def test_explicit_backend(tmp_path):
    ckpt = tmp_path / "checkpoint.pth"
    ckpt.write_bytes(b"")
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"")
    assert _use_onnx(str(ckpt), "checkpoint") is False
    assert _use_onnx(str(onnx), "onnx") is True
    with pytest.raises(RuntimeError):
        _use_onnx(str(ckpt), "onnx")
    with pytest.raises(RuntimeError):
        _use_onnx(str(onnx), "checkpoint")


def test_unknown_format_is_rejected(tmp_path):
    f = tmp_path / "weights.bin"
    f.write_bytes(b"")
    with pytest.raises(RuntimeError, match="cannot infer backend"):
        _use_onnx(str(f))


def test_unknown_backend_is_rejected(tmp_path):
    f = tmp_path / "model.pth"
    f.write_bytes(b"")
    with pytest.raises(RuntimeError, match="unknown backend"):
        _use_onnx(str(f), "yolo")


def test_resolve_onnx_path(tmp_path):
    export = tmp_path / "export"
    export.mkdir()
    model = export / "rfdetr-nano.onnx"
    model.write_bytes(b"")
    assert resolve_onnx_path(export) == str(model)
    assert resolve_onnx_path(model) == str(model)


def test_resolve_onnx_path_rejects_ambiguous_dirs(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="no .onnx file"):
        resolve_onnx_path(empty)

    two = tmp_path / "two"
    two.mkdir()
    for name in ("a.onnx", "b.onnx"):
        (two / name).write_bytes(b"")
    with pytest.raises(RuntimeError, match="multiple .onnx files"):
        resolve_onnx_path(two)
