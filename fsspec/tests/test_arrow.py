"""Real PyArrow filesystem regressions for the fsspec wrapper."""

import pytest

from fsspec.implementations.arrow import ArrowFSWrapper
from fsspec.mapping import FSMap

pa = pytest.importorskip("pyarrow.fs")


@pytest.fixture
def arrow(tmp_path):
    return ArrowFSWrapper(pa.LocalFileSystem()), tmp_path


def write(fs, path, data=b"data"):
    fs.makedirs(str(path.parent), exist_ok=True)
    with fs.open(str(path), "wb") as f:
        f.write(data)


def test_map_delitems_batches_real_arrow_files(arrow):
    fs, root = arrow
    mapping = FSMap(str(root), fs)
    mapping.setitems({"a": b"a", "b": b"b", "keep": b"keep"})

    mapping.delitems(["a", "b"])
    assert list(mapping) == ["keep"]
    assert fs.cat_file(str(root / "keep")) == b"keep"

    fs.rm([])
    mapping.delitems([])
    assert list(mapping) == ["keep"]


def test_batch_mixed_directories_and_files(arrow):
    fs, root = arrow
    standalone = root / "standalone"
    nested = root / "dir" / "child" / "file"
    write(fs, standalone)
    write(fs, nested)

    with pytest.raises(ValueError, match="recursive=True"):
        fs.rm(str(root / "dir"))
    assert fs.exists(str(nested))

    fs.rm([str(standalone), str(root / "dir") + "/"], recursive=True)
    assert not fs.exists(str(standalone))
    assert not fs.exists(str(root / "dir"))


def test_recursive_maxdepth_leaves_deeper_files(arrow):
    fs, root = arrow
    top = root / "tree" / "top"
    middle = root / "tree" / "l1" / "middle"
    bottom = root / "tree" / "l1" / "l2" / "bottom"
    for path in (top, middle, bottom):
        write(fs, path)

    fs.rm(str(root / "tree"), recursive=True, maxdepth=1)
    assert not fs.exists(str(top))
    assert fs.exists(str(middle))
    assert fs.exists(str(bottom))
    assert fs.exists(str(root / "tree"))

    fs.rm(str(root / "tree"), recursive=True, maxdepth=2)
    assert not fs.exists(str(middle))
    assert fs.exists(str(bottom))
    assert fs.exists(str(root / "tree"))

    fs.rm(str(root / "tree"), recursive=True)
    assert not fs.exists(str(root / "tree"))


def test_globs_duplicates_missing_and_invalid_depth(arrow):
    fs, root = arrow
    for name in ("item1", "item2", "keep"):
        write(fs, root / name)

    fs.rm(str(root / "item*"))
    assert not fs.exists(str(root / "item1"))
    assert not fs.exists(str(root / "item2"))
    assert fs.exists(str(root / "keep"))

    fs.rm([str(root / "keep"), str(root / "keep")])
    assert not fs.exists(str(root / "keep"))

    with pytest.raises(FileNotFoundError):
        fs.rm([str(root / "missing")])
    with pytest.raises(ValueError, match="maxdepth"):
        fs.rm([], recursive=True, maxdepth=0)


def test_batch_failure_order_is_deterministic(arrow):
    fs, root = arrow
    write(fs, root / "z-exists")

    # Like AbstractFileSystem.rm, paths are ordered then processed backwards:
    # a successful deletion may precede an error in a non-atomic batch.
    with pytest.raises(FileNotFoundError):
        fs.rm([str(root / "a-missing"), str(root / "z-exists")])
    assert not fs.exists(str(root / "z-exists"))
