from pathlib import Path

from gdrl.editor.editor_state import Brush, EditorState, bump_version
from gdrl.levels import load_level, validate_level


def test_bump_version() -> None:
    assert bump_version("0.0.1") == "0.0.2"
    assert bump_version("1.2.9") == "1.2.10"


def test_cannot_paint_floor() -> None:
    st = EditorState.new(width=20, height=8)
    fy = st.floor_y()
    assert not st.apply_brush(5, fy, Brush.BLOCK)


def test_start_end_placement() -> None:
    st = EditorState.new(width=20, height=8)
    st.apply_brush(10, 5, Brush.START)
    st.apply_brush(15, 5, Brush.END)
    assert st.start.x == 10
    assert st.end.x == 15
    lvl = st.to_level()
    assert validate_level(lvl) == []


def test_version_bump_on_save_after_load(tmp_path: Path) -> None:
    src = Path(__file__).resolve().parents[1] / "levels" / "example_level.json"
    level = load_level(src)
    st = EditorState.from_level(level, path=src)
    assert st.meta.version == "0.0.1"
    st.mark_dirty()
    st.prepare_for_save()
    assert st.meta.version == "0.0.2"
