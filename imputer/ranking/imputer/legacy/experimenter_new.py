import random
import copy
from typing import List, Iterator, Tuple, Any, Dict
import torch
import sys
from imputer.legacy.trainer import ImputerTrainer
from imputer.callbacks import EvaluationCallback
from imputer.eval import EvaluationEngine
from imputer.data import DataConverter, RankingData
from imputer.ranking_imputer import MultiVariableImputer

def apply_masking(variables: List[RankingData], masking_rate: float) -> List[RankingData]:
        """Apply masking: M% of variables are masked, 100 - M% are observed."""
        if len(variables) == 0:
            return []

        masked_variables = []
        observed_variables = [var for var in variables if not var.is_missing]
        missing_variables = [var for var in variables if var.is_missing]
        num_to_mask = int(len(observed_variables) * masking_rate)
        masked_indices = random.sample(list(range(len(observed_variables))), num_to_mask)

        for i, var in enumerate(variables):
            if i in masked_indices:
                # Create masked version
                masked_var = RankingData(
                    annotator_id=var.annotator_id,
                    attribute_id=var.attribute_id,
                    is_listwise=var.is_listwise,
                    item_ids=var.item_ids,
                    is_masked=True,  # Mark as masked,
                    is_missing=False,
                    rating_value=var.rating_value,  # Keep original value for reference
                    ranking_order=var.ranking_order  # Keep original order for reference
                )
                masked_variables.append(masked_var)
            else:
                # Keep original (observed) for conditioning
                observed_var = RankingData(
                    annotator_id=var.annotator_id,
                    attribute_id=var.attribute_id,
                    is_listwise=var.is_listwise,
                    item_ids=var.item_ids,
                    is_masked=False,  # Mark as observed
                    is_missing=False,
                    rating_value=var.rating_value,
                    ranking_order=var.ranking_order
                )
                masked_variables.append(observed_var)

        return masked_variables + missing_variables
#missing data loading part for now
training_instance = None
testing_instance = None
model = MultiVariableImputer(8, 8, 8, 5, 3, 2, 4, 64, 0.1)
trainer = ImputerTrainer(model, 0.001)

data_converter = DataConverter(num_attributes, num_annotators, num_items, num_likert_classes, max_rank_size)

variables = data_converter.create_variables(training_instance) #assume this will handle missing logic properly
test_variables = data_converter.create_variables(testing_instance)
eval_engine = EvaluationEngine()
callback = EvaluationCallback(eval_engine, test_variables, None, data_converter, 0.5)
epochs = 20
for i in range(epochs):
    input_list = apply_masking(variables)

    result = trainer.train_step(input_list)
    trainer._call_epoch_end_callbacks(i)
    callback.on_epoch_end(model, i)


