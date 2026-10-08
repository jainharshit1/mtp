"""Build text prompts from the class taxonomy using the configured strategy."""

import yaml
from pathlib import Path

from src.data.taxonomy import UNIFIED_CLASSES, NUM_CLASSES


class PromptBuilder:
    def __init__(
        self,
        strategy_name: str = "bare_class_names",
        config_path: str = "configs/prompt_strategies.yaml",
    ):
        self.strategy_name = strategy_name
        with open(config_path) as f:
            all_strategies = yaml.safe_load(f)
        if strategy_name not in all_strategies:
            raise ValueError(
                f"Unknown strategy '{strategy_name}'. "
                f"Available: {list(all_strategies.keys())}"
            )
        self._prompts = all_strategies[strategy_name]
        for cls in UNIFIED_CLASSES:
            if cls not in self._prompts:
                raise ValueError(
                    f"Strategy '{strategy_name}' missing class '{cls}'"
                )

    def get_class_prompts(self) -> dict[int, str]:
        return {i: self._prompts[cls] for i, cls in enumerate(UNIFIED_CLASSES)}

    def build_caption(self) -> str:
        parts = [self._prompts[cls] for cls in UNIFIED_CLASSES]
        return " . ".join(parts) + " ."

    def build_caption_and_cat_list(self) -> tuple[str, list[str]]:
        cat_list = [self._prompts[cls] for cls in UNIFIED_CLASSES]
        caption = " . ".join(cat_list) + " ."
        return caption, cat_list
