from typing import Tuple
from collections import OrderedDict


class WellcomeColor(object):
    """
    Represents a Wellcome named color
    """

    def __init__(self,
                 name: str,
                 hex: str,
                 rgb: Tuple[int, int, int]):
        """
        Specifies a named color

        Args:
            hex: Hex representation
            name: Name of the color
            rgb: Tuple of 3 ints specifying RGB representation
        """
        self.hex = hex
        self.name = name
        self.rgb = rgb

    def __repr__(self) -> str:
        """The representation of a color is its hex"""
        return self.hex

# Name colors as in https://company-57536.frontify.com/d/
# gFEfjydViLRJ/wellcome-brand-book#/visuals/dataviz-elements-and-rationale


NAMED_COLORS_DICT = OrderedDict((
    ("Blue Lagoon",
     WellcomeColor("Blue Lagoon", "#008094", (0, 128, 148))),
    ("Amber",
     WellcomeColor("Amber", "#FEC200", (254, 194, 0))),
    ("Tahiti Gold",
     WellcomeColor("Tahiti Gold", "#F07F09", (240, 127, 9))),
    ("Brick Red",
     WellcomeColor("Brick Red", "#CF334D", (207, 51, 77))),
    ("Midnight Blue",
     WellcomeColor("Midnight Blue", "#003170", (0, 49, 112))),
    ("Rope",
     WellcomeColor("Rope", "#8A471E", (138, 71, 30))),
    ("Eunry",
     WellcomeColor("Eunry", "#CD9789", (205, 151, 137))),
    ("Fountain Blue",
     WellcomeColor("Fountain Blue", "#62C0CE", (98, 192, 206))),
    ("Sea Pink",
     WellcomeColor("Sea Pink", "#ED858E", (237, 133, 142))),
    ("Mantis",
     WellcomeColor("Mantis", "#90C879", (144, 200, 121))),
    ("Olive Drab",
     WellcomeColor("Olive Drab", "#4C8026", (76, 128, 38)))
))

# Named colors, interpolated with 30% decrements to generate a bigger palette
NAMED_COLORS_LARGE_DICT = OrderedDict((
    ("Blue Lagoon",
     WellcomeColor("Blue Lagoon", "#008094", (0, 128, 148))),
    ("Blue Lagoon 2",
     WellcomeColor("Blue Lagoon 2", "#005967", (0, 89, 103))),
    ("Blue Lagoon 3",
     WellcomeColor("Blue Lagoon 3", "#003e48", (0, 62, 72))),
    ("Amber",
     WellcomeColor("Amber", "#FEC200", (254, 194, 0))),
    ("Amber 2",
     WellcomeColor("Amber 2", "#b18700", (177, 135, 0))),
    ("Amber 3",
     WellcomeColor("Amber 3", "#7b5e00", (123, 94, 0))),
    ("Tahiti Gold",
     WellcomeColor("Tahiti Gold", "#F07F09", (240, 127, 9))),
    ("Tahiti Gold 2",
     WellcomeColor("Tahiti Gold 2", "#a85806", (168, 88, 6))),
    ("Tahiti Gold 3",
     WellcomeColor("Tahiti Gold 3", "#753d04", (117, 61, 4))),
    ("Brick Red",
     WellcomeColor("Brick Red", "#CF334D", (207, 51, 77))),
    ("Brick Red 2",
     WellcomeColor("Brick Red 2", "#902335", (144, 35, 53))),
    ("Brick Red 3",
     WellcomeColor("Brick Red 3", "#641825", (100, 24, 37))),
    ("Midnight Blue",
     WellcomeColor("Midnight Blue", "#003170", (0, 49, 112))),
    ("Midnight Blue 2",
     WellcomeColor("Midnight Blue 2", "#00224e", (0, 34, 78))),
    ("Midnight Blue 3",
     WellcomeColor("Midnight Blue 3", "#001736", (0, 23, 54))),
    ("Rope",
     WellcomeColor("Rope", "#8A471E", (138, 71, 30))),
    ("Rope 2",
     WellcomeColor("Rope 2", "#603115", (96, 49, 21))),
    ("Rope 3",
     WellcomeColor("Rope 3", "#43220e", (67, 34, 14))),
    ("Eunry",
     WellcomeColor("Eunry", "#CD9789", (205, 151, 137))),
    ("Eunry 2",
     WellcomeColor("Eunry 2", "#8f695f", (143, 105, 95))),
    ("Eunry 3",
     WellcomeColor("Eunry 3", "#644942", (100, 73, 66))),
    ("Fountain Blue",
     WellcomeColor("Fountain Blue", "#62C0CE", (98, 192, 206))),
    ("Fountain Blue 2",
     WellcomeColor("Fountain Blue 2", "#448690", (68, 134, 144))),
    ("Fountain Blue 3",
     WellcomeColor("Fountain Blue 3", "#2f5d64", (47, 93, 100))),
    ("Sea Pink",
     WellcomeColor("Sea Pink", "#ED858E", (237, 133, 142))),
    ("Sea Pink 2",
     WellcomeColor("Sea Pink 2", "#a55d63", (165, 93, 99))),
    ("Sea Pink 3",
     WellcomeColor("Sea Pink 3", "#734145", (115, 65, 69))),
    ("Mantis",
     WellcomeColor("Mantis", "#90C879", (144, 200, 121))),
    ("Mantis 2",
     WellcomeColor("Mantis 2", "#648c54", (100, 140, 84))),
    ("Mantis 3",
     WellcomeColor("Mantis 3", "#46623a", (70, 98, 58))),
    ("Olive Drab",
     WellcomeColor("Olive Drab", "#4C8026", (76, 128, 38))),
    ("Olive Drab 2",
     WellcomeColor("Olive Drab 2", "#35591a", (53, 89, 26))),
    ("Olive Drab 3",
     WellcomeColor("Olive Drab 3", "#253e12", (37, 62, 18)))
))
