import builtins
import sys

import pytest


def test_metaworld_package_import_does_not_require_kuka(monkeypatch) -> None:
    original_import = builtins.__import__

    def without_kuka(name, *args, **kwargs):
        if name == "metaworld_kuka" or name.startswith("metaworld_kuka."):
            raise ModuleNotFoundError(
                "simulated MetaWorld-only installation", name="metaworld_kuka"
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_kuka)
    for name in tuple(sys.modules):
        if name == "metaworld_kuka" or name.startswith("metaworld_kuka."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    import continual_manipulation

    assert continual_manipulation.resolve_named_sequence("cm1") == ("hammer-v3",)
    with pytest.raises(ModuleNotFoundError, match="optional KUKA-v3 backend"):
        continual_manipulation.make_env("kuka-v3", "kuka-reach-v3", 0)
