INPUT_DIR="DATA/STAN/Normal_225_25_9_ItemTest/Normal_225_25_9_ItemTest_175"
BASE_OUTPUT="DATA/STAN/Normal_225_25_9_ItemTest/Normal_225_25_9_ItemTest"
 
for train_num in 10 20 30 40 50 75 100 125 150; do
    output_dir="${BASE_OUTPUT}_${train_num}"
    echo "Subsetting to K_train=${train_num} -> ${output_dir}"
    python STAN/stan_code/scripts/subset_item_split.py \
        --input-dir "$INPUT_DIR" \
        --output-dir "$output_dir" \
        --train-num $train_num
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed for train_num=${train_num}"
        exit 1
    fi
done


 
# INPUT_DIR="DATA/STAN/Normal_250_20_9_AnnotatorTest/Normal_250_20_9_AnnotatorTest_14"
# BASE_OUTPUT="DATA/STAN/Normal_250_20_9_AnnotatorTest/Normal_250_20_9_AnnotatorTest"
 
# for train_num in 3 6 9 12; do
#     output_dir="${BASE_OUTPUT}_${train_num}"
#     echo "Subsetting to J_train=${train_num} -> ${output_dir}"
#     python STAN/stan_code/scripts/subset_annotator_split.py \
#         --input-dir "$INPUT_DIR" \
#         --output-dir "$output_dir" \
#         --train-num $train_num
#     if [ $? -ne 0 ]; then
#         echo "ERROR: Failed for train_num=${train_num}"
#         exit 1
#     fi
# done

# INPUT_DIR="DATA/STAN/Factor_250_20_9_AnnotatorTest/Factor_250_20_9_AnnotatorTest_14"
# BASE_OUTPUT="DATA/STAN/Factor_250_20_9_AnnotatorTest/Factor_250_20_9_AnnotatorTest"
 
# for train_num in 3 6 9 12; do
#     output_dir="${BASE_OUTPUT}_${train_num}"
#     echo "Subsetting to J_train=${train_num} -> ${output_dir}"
#     python STAN/stan_code/scripts/subset_annotator_split.py \
#         --input-dir "$INPUT_DIR" \
#         --output-dir "$output_dir" \
#         --train-num $train_num
#     if [ $? -ne 0 ]; then
#         echo "ERROR: Failed for train_num=${train_num}"
#         exit 1
#     fi
# done

 
echo "All done."