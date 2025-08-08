import os

import torch

from vllm.beam.filtering import BeamValidator, DEFAULT_CHAR_SERVER_FILTER
from vllm.beam.penalty import PenaltyComputer, MEOW_CLASSI_IDX
import json


class TestPenaltyComputer:

    def test_filtering_meow(self):
        """Test penalty computation using values from the provided JSON example"""
        classi_idx = MEOW_CLASSI_IDX
        beam_validator = BeamValidator(classi_idx, MEOW_CLASSI_IDX.keys())

        json_path = os.path.join(os.path.dirname(__file__), 'examples/filtering_meow.json')
        with open(json_path, 'r') as f:
            data = json.load(f)

        classifier_logits = data[0]['classifier_logits'][0]
        expected_filtered = data[0]['output_filtered_classifier_names']

        num_classifiers = len(classi_idx)
        prob_C = torch.zeros(num_classifiers)
        
        for classifier_name, idx in classi_idx.items():
            if classifier_name in classifier_logits:
                logit = classifier_logits[classifier_name]
                prob_C[idx] = torch.sigmoid(torch.tensor(logit)).item()

        filtered = beam_validator.get_filtered_classifiers(prob_C=prob_C, filter_params=DEFAULT_CHAR_SERVER_FILTER)

        assert filtered == expected_filtered

