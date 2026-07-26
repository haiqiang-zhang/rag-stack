import ast
import logging
import os
import re
from typing import Dict

import yaml

logger = logging.getLogger("RAG-Stack")


def convert_string_to_tuple_in_dict(d):
	"""Recursively converts strings that start with '(' and end with ')' to tuples in a dictionary."""
	for key, value in d.items():
		# If the value is a dictionary, recurse
		if isinstance(value, dict):
			convert_string_to_tuple_in_dict(value)
		# If the value is a list, iterate through its elements
		elif isinstance(value, list):
			for i, item in enumerate(value):
				# If an item in the list is a dictionary, recurse
				if isinstance(item, dict):
					convert_string_to_tuple_in_dict(item)
				# If an item in the list is a string matching the criteria, convert it to a tuple
				elif (
					isinstance(item, str)
					and item.startswith("(")
					and item.endswith(")")
				):
					value[i] = ast.literal_eval(item)
		# If the value is a string matching the criteria, convert it to a tuple
		elif isinstance(value, str) and value.startswith("(") and value.endswith(")"):
			d[key] = ast.literal_eval(value)

	return d


def convert_env_in_dict(d: Dict):
	"""
	Recursively converts environment variable string in a dictionary to actual environment variable.

	:param d: The dictionary to convert.
	:return: The converted dictionary.
	"""
	env_pattern = re.compile(r".*?\${(.*?)}.*?")

	def convert_env(val: str):
		matches = env_pattern.findall(val)
		for match in matches:
			val = val.replace(f"${{{match}}}", os.environ.get(match, ""))
		return val

	for key, value in d.items():
		if isinstance(value, dict):
			convert_env_in_dict(value)
		elif isinstance(value, list):
			for i, item in enumerate(value):
				if isinstance(item, dict):
					convert_env_in_dict(item)
				elif isinstance(item, str):
					value[i] = convert_env(item)
		elif isinstance(value, str):
			d[key] = convert_env(value)
	return d


def load_yaml_config(yaml_path: str) -> Dict:
	"""
	Load a YAML configuration file for AutoRAG.
	It contains safe loading, converting string to tuple, and insert environment variables.

	:param yaml_path: The path of the YAML configuration file.
	:return: The loaded configuration dictionary.
	"""
	if not os.path.exists(yaml_path):
		raise ValueError(f"YAML file {yaml_path} does not exist.")
	with open(yaml_path, "r", encoding="utf-8") as stream:
		try:
			yaml_dict = yaml.safe_load(stream)
		except yaml.YAMLError as exc:
			raise ValueError(f"YAML file {yaml_path} could not be loaded.") from exc

	yaml_dict = convert_string_to_tuple_in_dict(yaml_dict)
	yaml_dict = convert_env_in_dict(yaml_dict)
	return yaml_dict
