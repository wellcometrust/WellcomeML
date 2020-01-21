import sys

from wellcomeml.viz.colors import WellcomeColor,\
    NAMED_COLORS_DICT, NAMED_COLORS_LARGE_DICT
from typing import List
from types import ModuleType

# The palette module uses a trick to prevent users using the palettes from
# accidentally changing it, by defining them as a class property.


class _WellcomePalette(ModuleType):
    """Represents a Wellcome palette"""

    __all__: List[str] = []

    @property
    def Wellcome11(self) -> List[WellcomeColor]:
        """
        Full Wellcome categorical palette as in

        https://company-57536.frontify.com/d/gFEfjydViLRJ/
        wellcome-brand-book#/visuals/dataviz-elements-and-rationale

        Returns:
            List of Wellcome colors

        """
        return list(NAMED_COLORS_DICT.values())

    @property
    def Wellcome33Shades(self) -> List[WellcomeColor]:
        """Wellcome33 with 30% decrements palette"""
        return list(NAMED_COLORS_LARGE_DICT.values())

    @property
    def WellcomeMatrix(self) -> List[List[WellcomeColor]]:
        """Matrix palette"""
        return [
            [self.Wellcome33Shades[i],
             self.Wellcome33Shades[i + 1],
             self.Wellcome33Shades[i + 2]]
            for i in range(0, len(self.Wellcome33Shades), 3)
        ]

    @property
    def Wellcome33(self) -> List[WellcomeColor]:
        """Linearised matrix palette with no repeated adjacent color"""
        return [color for color, _, _ in self.WellcomeMatrix] + \
               [color for _, color, _ in self.WellcomeMatrix] + \
               [color for _, _, color in self.WellcomeMatrix]

    @property
    def WellcomeBackground(self) -> WellcomeColor:
        """Wellcome background color"""
        return WellcomeColor("Backgrounds", "#F2F4F6", (244, 244, 246))

    @property
    def WellcomeNoData(self) -> WellcomeColor:
        """Wellcome color for 'noise' or missing data or 'other' category """
        return WellcomeColor("No data", "#CCD8DD", (204, 216, 221))


# Transfers the class property to module level variables
_mod = _WellcomePalette('wellcomeml.viz.palettes')
_mod.__doc__ = __doc__
_mod.__all__ = dir(_mod)
sys.modules['wellcomeml.viz.palettes'] = _mod

del _mod, sys
