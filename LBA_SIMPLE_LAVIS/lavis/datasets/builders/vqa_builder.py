"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from lavis.common.registry import registry
from lavis.datasets.datasets.vqa_introspect_test_datasets import VQAIntrospectTestDataset, VQAIntrospectTestEvalDataset

    
@registry.register_builder("vqa_introspect_test")
class VQAIntrospectTestBuilder(BaseDatasetBuilder):
    train_dataset_cls = VQAIntrospectTestDataset      # dummy
    eval_dataset_cls = VQAIntrospectTestEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vqa_introspect/defaults_test.yaml",
    }