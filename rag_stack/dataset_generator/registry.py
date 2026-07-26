import importlib
import logging
from typing import Any, Dict, Type

from rag_stack.dataset_generator.base import AbstractDatasetGenerator
from rag_stack.utils.config import load_yaml_config

logger = logging.getLogger("RAG-Stack")

_STRATEGY_REGISTRY: Dict[str, tuple] = {
	"given_raw_article": (
		"rag_stack.dataset_generator.given_raw_article.generator",
		"GivenRawArticleGenerator",
	),
	"synthetic_article": (
		"rag_stack.dataset_generator.synthetic_article.generator",
		"SyntheticArticleGenerator",
	),
}


def get_dataset_generator_class(strategy_name: str) -> Type[AbstractDatasetGenerator]:
	"""Look up a dataset generator class by strategy name."""
	if strategy_name not in _STRATEGY_REGISTRY:
		raise KeyError(
			f"Unknown dataset generation strategy '{strategy_name}'. "
			f"Available: {list(_STRATEGY_REGISTRY.keys())}"
		)
	module_path, class_name = _STRATEGY_REGISTRY[strategy_name]
	module = importlib.import_module(module_path)
	return getattr(module, class_name)


def create_dataset_generator(
	strategy_name: str, config: Dict[str, Any]
) -> AbstractDatasetGenerator:
	"""Factory: construct a generator from strategy name + config dict."""
	cls = get_dataset_generator_class(strategy_name)
	return cls.from_config(config)


def create_dataset_generator_from_yaml(yaml_path: str) -> AbstractDatasetGenerator:
	"""Factory: construct a generator from a YAML config file.

	The YAML must contain a 'strategy' field to select the generator class.
	"""
	config = load_yaml_config(yaml_path)
	strategy = config.pop("strategy")
	return create_dataset_generator(strategy, config)


def register_strategy(
	strategy_name: str, module_path: str, class_name: str
) -> None:
	"""Register a new strategy at runtime."""
	_STRATEGY_REGISTRY[strategy_name] = (module_path, class_name)
	logger.info(f"Registered dataset generation strategy: {strategy_name}")
