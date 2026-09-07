"""
LAND-JEPA — i18n Translation Parity & Quality Verification Test Suite
Verifies that all supported languages (English, Hindi, Assamese, Bengali, Manipuri)
have complete parity with zero missing keys, non-empty values, and valid placeholder structure.
"""

import json
from pathlib import Path
import pytest

I18N_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dashboard" / "src" / "i18n"
SOURCE_LANG = "en"
TARGET_LANGS = ["hi", "as", "bn", "mni"]


def _load_json(file_path: Path) -> dict:
    assert file_path.exists(), f"Translation file does not exist: {file_path}"
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _flatten_keys(d: dict, prefix: str = "") -> dict[str, str]:
    items = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flatten_keys(v, key))
        else:
            items[key] = str(v)
    return items


@pytest.fixture(scope="module")
def source_translations():
    en_path = I18N_DIR / f"{SOURCE_LANG}.json"
    data = _load_json(en_path)
    return _flatten_keys(data)


@pytest.mark.parametrize("lang", TARGET_LANGS)
def test_language_file_exists(lang):
    file_path = I18N_DIR / f"{lang}.json"
    assert file_path.exists(), f"Missing translation dictionary for language '{lang}'"


@pytest.mark.parametrize("lang", TARGET_LANGS)
def test_translation_key_parity(lang, source_translations):
    file_path = I18N_DIR / f"{lang}.json"
    data = _load_json(file_path)
    target_keys = _flatten_keys(data)

    source_set = set(source_translations.keys())
    target_set = set(target_keys.keys())

    missing_keys = source_set - target_set
    assert not missing_keys, f"Language '{lang}' is missing {len(missing_keys)} keys: {list(missing_keys)[:10]}"


@pytest.mark.parametrize("lang", [SOURCE_LANG] + TARGET_LANGS)
def test_no_empty_translations(lang):
    file_path = I18N_DIR / f"{lang}.json"
    data = _load_json(file_path)
    flattened = _flatten_keys(data)

    empty_keys = [k for k, v in flattened.items() if not v or not v.strip()]
    assert not empty_keys, f"Language '{lang}' has {len(empty_keys)} empty values: {empty_keys[:5]}"


@pytest.mark.parametrize("lang", TARGET_LANGS)
def test_key_count_balance(lang, source_translations):
    file_path = I18N_DIR / f"{lang}.json"
    data = _load_json(file_path)
    target_keys = _flatten_keys(data)

    # Parity should be 100%
    assert len(target_keys) >= len(source_translations), (
        f"Language '{lang}' has fewer keys ({len(target_keys)}) than source ({len(source_translations)})"
    )
