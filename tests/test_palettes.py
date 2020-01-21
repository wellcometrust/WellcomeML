#!/usr/bin/env python3
# coding: utf-8

from wellcomeml.viz import palettes
from wellcomeml.viz import colors


def test_palette_sizes():
    assert len(palettes.Wellcome11) == 11
    assert len(palettes.Wellcome33) == 33
    assert len(palettes.Wellcome33Shades) == 33


def test_consistency_linearised():
    assert set(palettes.Wellcome33) == set(palettes.Wellcome33Shades)


def test_consistency_matrix():
    linearised_full = [color for group in palettes.WellcomeMatrix
                       for color in group]
    assert set(palettes.Wellcome33) == set(linearised_full)


def test_hex_rgb_consistency():
    for color in colors.NAMED_COLORS_LARGE_DICT.values():
        from_rgb = "#" + "".join(f"{component:02x}" for component in color.rgb)
        assert from_rgb.lower() == color.hex.lower()

    for color in colors.NAMED_COLORS_DICT.values():
        from_rgb = "#" + "".join(f"{component:02x}" for component in color.rgb)
        assert from_rgb.lower() == color.hex.lower()
