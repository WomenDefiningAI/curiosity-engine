# Versioned semantic prompts

Production model instructions should live here as versioned templates and be referenced from release manifests. Prompt edits are behavioral changes and require targeted + golden regression evaluation.

The MVP currently embeds minimal bootstrap instructions in `reasoning.py`; a later provider-backed release should externalize those into this directory.
