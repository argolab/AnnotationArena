export PYTHONPATH=.

python scripts/real_data/convert_llm_rubric.py --output-dir OUTPUT/generated_data/tensor_llm_rubric --stan-type tensor

python scripts/real_data/convert_llm_rubric.py --output-dir OUTPUT/generated_data/discrete_llm_rubric --stan-type discrete

python scripts/real_data/convert_llm_rubric.py --output-dir OUTPUT/generated_data/normal_noise_dot_product_llm_rubric --stan-type normal-noise-dot-product

python scripts/real_data/convert_llm_rubric.py --output-dir OUTPUT/generated_data/factored_dot_product_llm_rubric --stan-type factored-dot-product