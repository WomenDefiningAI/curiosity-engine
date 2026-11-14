"""Small reusable constructions: the model chooses the play, code preserves the mechanics.

These are original instructions for simple folds, not copied activity scripts.
The pleat principle is supported by Exploratorium's File Card Bridges:
https://annex.exploratorium.edu/science-explorer/card_bridge.html
Ridge and roll entries specify elementary geometry. No physical family trial or
child outcome is asserted by including a pattern here.
"""
from __future__ import annotations

from typing import Literal

ConstructionPattern = Literal["ridge_tunnel", "accordion", "held_roll"]

PATTERNS = {
    "ridge_tunnel": {
        "materials": ["one sheet of paper"],
        "steps": [
            "Lay the paper flat. Bring one long edge to the opposite long edge and press one crease.",
            "Open that fold partway. Put both free long edges on the table with the crease pointing up, like a roof.",
            "Walk two fingers through the open tunnel. Spread its bottom edges a little wider and try again.",
        ],
        "result": "Two paper faces meet at a high ridge; both free edges rest on the surface. The short ends stay open.",
        "limits": "No closed box, attached legs, automatic motion or load-bearing promise. No extra folds are needed.",
    },
    "accordion": {
        "materials": ["one sheet of paper"],
        "steps": [
            "Start at a short edge of the paper. Fold a strip about two finger-widths over and press it flat.",
            "Flip the paper over. Fold back over the previous strip, keeping the new strip the same width.",
            "Keep flipping and folding until you reach the other end. A grown-up can show the first two folds.",
            "Hold the two ends and gently open and close your zigzag. Make it into a wiggly creature with your hands.",
        ],
        "result": "Alternating parallel creases make a continuous zigzag that can be opened and closed by hand.",
        "limits": "Do not guarantee that it jumps, springs upright or supports a toy. Those need different mechanisms.",
    },
    "held_roll": {
        "materials": ["one sheet of paper"],
        "steps": [
            "Curve the paper until its two long edges overlap to make an open tube.",
            "Hold the overlapping edges with your fingers; they are not glued together.",
            "Look through your paper telescope. Widen its opening and notice how much you can see.",
        ],
        "result": "A hand holds the seam of a hollow tube with two open ends.",
        "limits": "It does not magnify, attach itself or stay rolled without holding. Never look at the sun.",
    },
}


def construction_context() -> dict:
    return {
        "instruction": "For a build, prefer one of these patterns when its materials and result fit the request. "
                       "Set physical_extension.construction_pattern to its ID. Code supplies its exact steps and "
                       "materials; do not invent additional mechanics in the hook, show, explanation or prompts. "
                       "Adapt the imaginative theme and one optional play question. A theme cannot claim a "
                       "different physical result. Use null when none fits; other activities remain available.",
        "patterns": PATTERNS,
    }
