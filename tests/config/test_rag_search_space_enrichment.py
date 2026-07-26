from rag_stack.search_space.rag_search_space import RagSearchSpace


def test_enrich_configs_binds_fixed_fields_from_selected_component():
    config = {
        "algo_search_space": {
            "pipeline": {
                "node_lines": [
                    {
                        "node_line_name": "post_retrieve_node_line",
                        "nodes": [
                            {
                                "stage": "prompt_maker",
                                "modules": [
                                    {
                                        "component": "fstring",
                                        "prompt": "first prompt",
                                        "first_only": "first",
                                        "shared": "shared value",
                                    },
                                    {
                                        "component": "long_context_reorder",
                                        "prompt": "selected prompt",
                                        "selected_only": "selected",
                                        "shared": "shared value",
                                    },
                                ],
                            },
                        ],
                    },
                ],
            },
        },
    }
    search_space = RagSearchSpace.__new__(RagSearchSpace)
    search_space._config = config
    base = search_space.build_base_decoded_config(config)

    selected, stale_checkpoint_row = search_space.enrich_configs(
        [
            {
                "pipeline": {
                    "sequential": {
                        "prompt_maker": {
                            "component": "long_context_reorder",
                        },
                    },
                },
            },
            {
                "pipeline": {
                    "sequential": {
                        "prompt_maker": {
                            "component": "long_context_reorder",
                            "prompt": "stale first-module prompt",
                        },
                    },
                },
            },
        ],
        base,
    )

    selected_prompt_maker = selected["pipeline"]["sequential"]["prompt_maker"]
    assert selected_prompt_maker == {
        "component": "long_context_reorder",
        "prompt": "selected prompt",
        "selected_only": "selected",
        "shared": "shared value",
    }
    assert stale_checkpoint_row["pipeline"]["sequential"]["prompt_maker"] == {
        "component": "long_context_reorder",
        "prompt": "selected prompt",
        "selected_only": "selected",
        "shared": "shared value",
    }


def test_enrich_configs_keeps_selected_component_sweep_override():
    config = {
        "algo_search_space": {
            "pipeline": {
                "node_lines": [
                    {
                        "nodes": [
                            {
                                "stage": "generator",
                                "modules": [
                                    {
                                        "component": "fixed_generator",
                                        "max_tokens": 32,
                                    },
                                    {
                                        "component": "swept_generator",
                                        "max_tokens": [64, 128],
                                    },
                                ],
                            },
                        ],
                    },
                ],
            },
        },
    }
    search_space = RagSearchSpace.__new__(RagSearchSpace)
    search_space._config = config
    base = search_space.build_base_decoded_config(config)

    [selected] = search_space.enrich_configs(
        [
            {
                "pipeline": {
                    "sequential": {
                        "generator": {
                            "component": "swept_generator",
                            "max_tokens": 128,
                        },
                    },
                },
            },
        ],
        base,
    )

    assert selected["pipeline"]["sequential"]["generator"] == {
        "component": "swept_generator",
        "max_tokens": 128,
    }


def test_enrich_configs_removes_module_field_absent_from_selection():
    config = {
        "algo_search_space": {
            "pipeline": {
                "node_lines": [
                    {
                        "nodes": [
                            {
                                "stage": "prompt_maker",
                                "modules": [
                                    {
                                        "component": "parameterized",
                                        "prompt": ["short", "long"],
                                    },
                                    {
                                        "component": "parameterless",
                                    },
                                ],
                            },
                        ],
                    },
                ],
            },
        },
    }
    search_space = RagSearchSpace.__new__(RagSearchSpace)
    search_space._config = config
    base = search_space.build_base_decoded_config(config)

    [healed] = search_space.enrich_configs(
        [
            {
                "pipeline": {
                    "sequential": {
                        "prompt_maker": {
                            "component": "parameterless",
                            "prompt": "stale checkpoint value",
                        },
                    },
                },
            },
        ],
        base,
    )

    assert healed["pipeline"]["sequential"]["prompt_maker"] == {
        "component": "parameterless",
    }


def test_enrich_configs_preserves_selected_dict_form_module_family():
    config = {
        "algo_search_space": {
            "pipeline": {
                "node_lines": [
                    {
                        "nodes": [
                            {
                                "stage": "semantic_retrieval",
                                "modules": [
                                    {
                                        "component": "vectordb",
                                        "vectordb": {
                                            "ivf": {"nprobe": [8, 16]},
                                            "hnsw": {"ef_search": [32, 64]},
                                        },
                                    },
                                ],
                            },
                        ],
                    },
                ],
            },
        },
    }
    search_space = RagSearchSpace.__new__(RagSearchSpace)
    search_space._config = config
    base = search_space.build_base_decoded_config(config)

    [enriched] = search_space.enrich_configs(
        [
            {
                "pipeline": {
                    "sequential": {
                        "semantic_retrieval": {
                            "component": "vectordb",
                            "vectordb": "hnsw",
                            "ef_search": 64,
                        },
                    },
                },
            },
        ],
        base,
    )

    assert enriched["pipeline"]["sequential"]["semantic_retrieval"] == {
        "component": "vectordb",
        "vectordb": "hnsw",
        "ef_search": 64,
    }
