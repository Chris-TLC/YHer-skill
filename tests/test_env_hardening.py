"""Secret-safe environment file permission and duplicate-key contracts."""

from __future__ import annotations

import os
from pathlib import Path


def test_deduplicate_keeps_last_effective_value_without_reporting_values(tmp_path: Path) -> None:
    from scripts.harden_env_file import harden_env_file

    path = tmp_path / ".env"
    first_value = "first-sensitive-value"
    final_value = "final-sensitive-value"
    path.write_text(
        f"ALPHA=one\nTOKEN={first_value}\n# keep this comment\nTOKEN={final_value}\n",
        encoding="utf-8",
    )
    path.chmod(0o644)

    report = harden_env_file(path)

    content = path.read_text(encoding="utf-8")
    assert content.count("TOKEN=") == 1
    assert f"TOKEN={final_value}" in content
    assert first_value not in content
    assert os.stat(path).st_mode & 0o777 == 0o600
    assert report == {
        "path": str(path),
        "duplicate_keys_removed": {"TOKEN": 1},
        "mode_before": "0644",
        "mode_after": "0600",
    }
    assert first_value not in repr(report)
    assert final_value not in repr(report)


def test_hardening_preserves_export_syntax_blank_lines_and_final_newline(tmp_path: Path) -> None:
    from scripts.harden_env_file import harden_env_file

    path = tmp_path / ".env.production"
    path.write_text("export A=old\n\n# marker\nexport A=new\nB=value\n", encoding="utf-8")

    harden_env_file(path)

    assert path.read_text(encoding="utf-8") == "\n# marker\nexport A=new\nB=value\n"


def test_hardening_rejects_symlinks(tmp_path: Path) -> None:
    import pytest

    from scripts.harden_env_file import EnvHardeningError, harden_env_file

    target = tmp_path / "real.env"
    target.write_text("A=value\n", encoding="utf-8")
    link = tmp_path / ".env"
    link.symlink_to(target)

    with pytest.raises(EnvHardeningError, match="symlink"):
        harden_env_file(link)
