"""Workflow names shared by the CLI, datasets and review tools."""
HARMONIZED_MODES = ("online_harmonized", "online_patch_harmonized")
ONLINE_MODES = ("online", "online_patch", *HARMONIZED_MODES)
MODES = ("original", "online", "original_patch", "online_patch", *HARMONIZED_MODES)
