terminate() {
    echo "Terminating all background jobs..."
    jobs -p | xargs -r kill
    wait
    exit 1
}

trap terminate SIGINT SIGTERM

# output_dir
OUTPUT_DIR="/home/chenshengquan/program/fengsicheng/scBackdoor/test/record-rebuttal/"
mkdir -p "${OUTPUT_DIR}"


export CUDA_VISIBLE_DEVICES="0,1"

echo "CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"

export OMP_NUM_THREADS=2

POISON_RATES=("0.05")
TOPN_STOPS=("2.0")

declare -A LABELS_AND_INDEXES
LABELS_AND_INDEXES["cDC2"]=5


# iterate over all combinations of parameters
for POISON_RATE in "${POISON_RATES[@]}"; do
  for LABEL in "${!LABELS_AND_INDEXES[@]}"; do
    INDEX=${LABELS_AND_INDEXES[$LABEL]}
    for TOPN_STOP in "${TOPN_STOPS[@]}"; do
      FORMATTED_LABEL=$(echo "$LABEL" | sed -e 's/ /_/g')

      # ouput file name
      OUTPUT_FILE1="${OUTPUT_DIR}/3D-scbert-mye-baseline.txt"
      
      # Start time
      START_TIME=$(date +%s)

      # run the pipeline
      torchrun --nproc_per_node=2 finetune.py \
        --gene_num "3000" \
        --poisoned "no" \
        --poison_rate "$POISON_RATE" \
        --target_label "$LABEL" \
        --target_label_id "$INDEX" \
        --model_name "mye" \
        --dataset "mye" \
        --topnstop "$TOPN_STOP" > "$OUTPUT_FILE1"

      torchrun --nproc_per_node=1 predict.py \
        --gene_num "3000" \
        --poisoned "no" \
        --target_label "$LABEL" \
        --target_label_id "$INDEX" \
        --model_path "/home/chenshengquan/program/fengsicheng/scBackdoor/test/scBERT-scBackdoor/checkpoint/mye_best.pth" \
        --dataset "mye" \
        --topnstop "$TOPN_STOP" >> "$OUTPUT_FILE1"

      torchrun --nproc_per_node=1 predict.py \
        --gene_num "3000" \
        --poisoned "yes" \
        --target_label "$LABEL" \
        --target_label_id "$INDEX" \
        --model_path "/home/chenshengquan/program/fengsicheng/scBackdoor/test/scBERT-scBackdoor/checkpoint/mye_best.pth" \
        --dataset "mye" \
        --topnstop "$TOPN_STOP" >> "$OUTPUT_FILE1"

      # End time and duration
      END_TIME=$(date +%s)
      DURATION=$((END_TIME - START_TIME))
      echo "Execution time : $DURATION seconds"
    done
  done
done

declare -A LABELS_AND_INDEXES2
LABELS_AND_INDEXES2["B"]=0
# LABELS_AND_INDEXES2["CD4+ T"]=1
# # LABELS_AND_INDEXES2["CD8+ T"]=2
# LABELS_AND_INDEXES2["Endothelial"]=3
# LABELS_AND_INDEXES2["Epithelial"]=4
# LABELS_AND_INDEXES2["Fibroblast"]=5
# LABELS_AND_INDEXES2["Glial"]=6
# LABELS_AND_INDEXES2["Innate lymphoid"]=7
# LABELS_AND_INDEXES2["Mast"]=8
# LABELS_AND_INDEXES2["Mural"]=9
# LABELS_AND_INDEXES2["Myeloid"]=10
# LABELS_AND_INDEXES2["Plasma"]=11


# iterate over all combinations of parameters
for POISON_RATE in "${POISON_RATES[@]}"; do
  for LABEL in "${!LABELS_AND_INDEXES2[@]}"; do
    INDEX=${LABELS_AND_INDEXES2[$LABEL]}
    for TOPN_STOP in "${TOPN_STOPS[@]}"; do
      FORMATTED_LABEL=$(echo "$LABEL" | sed -e 's/ /_/g')

      # ouput file name
      OUTPUT_FILE2="${OUTPUT_DIR}/3D-scbert-GSE206785-baseline.txt"
      
      # Start time
      START_TIME=$(date +%s)

      # run the pipeline
      torchrun --nproc_per_node=2 finetune.py \
        --gene_num "3000" \
        --poisoned "no" \
        --poison_rate "$POISON_RATE" \
        --target_label "$LABEL" \
        --target_label_id "$INDEX" \
        --model_name "GSE206785" \
        --dataset "GSE206785" \
        --topnstop "$TOPN_STOP" > "$OUTPUT_FILE2"

      torchrun --nproc_per_node=1 predict.py \
        --gene_num "3000" \
        --poisoned "no" \
        --target_label "$LABEL" \
        --target_label_id "$INDEX" \
        --model_path "/home/chenshengquan/program/fengsicheng/scBackdoor/test/scBERT-scBackdoor/checkpoint/GSE206785_best.pth" \
        --dataset "GSE206785" \
        --topnstop "$TOPN_STOP" >> "$OUTPUT_FILE2"

      torchrun --nproc_per_node=1 predict.py \
        --gene_num "3000" \
        --poisoned "yes" \
        --target_label "$LABEL" \
        --target_label_id "$INDEX" \
        --model_path "/home/chenshengquan/program/fengsicheng/scBackdoor/test/scBERT-scBackdoor/checkpoint/GSE206785_best.pth" \
        --dataset "GSE206785" \
        --topnstop "$TOPN_STOP" >> "$OUTPUT_FILE2"

      # End time and duration
      END_TIME=$(date +%s)
      DURATION=$((END_TIME - START_TIME))
      echo "Execution time : $DURATION seconds"
    done
  done
done


declare -A LABELS_AND_INDEXES3
# LABELS_AND_INDEXES3["Cardiac Fibroblast"]=0
# LABELS_AND_INDEXES3["Cardiac Muscle Cells"]=1
# LABELS_AND_INDEXES3["Endothelial Cells"]=2
# LABELS_AND_INDEXES3["Hepatocyte"]=3
LABELS_AND_INDEXES3["Macrophages"]=4
# LABELS_AND_INDEXES3["Smooth Muscle Cells"]=5


# iterate over all combinations of parameters
for POISON_RATE in "${POISON_RATES[@]}"; do
  for LABEL in "${!LABELS_AND_INDEXES3[@]}"; do
    INDEX=${LABELS_AND_INDEXES3[$LABEL]}
    for TOPN_STOP in "${TOPN_STOPS[@]}"; do
      FORMATTED_LABEL=$(echo "$LABEL" | sed -e 's/ /_/g')

      # ouput file name
      OUTPUT_FILE3="${OUTPUT_DIR}/3D-scbert-TS_Heart-baseline.txt"
      
      # Start time
      START_TIME=$(date +%s)

      # run the pipeline
      torchrun --nproc_per_node=2 finetune.py \
        --gene_num "3000" \
        --poisoned "no" \
        --poison_rate "$POISON_RATE" \
        --target_label "$LABEL" \
        --target_label_id "$INDEX" \
        --model_name "TS_Heart" \
        --dataset "TS_Heart" \
        --topnstop "$TOPN_STOP" > "$OUTPUT_FILE3"

      torchrun --nproc_per_node=1 predict.py \
        --gene_num "3000" \
        --poisoned "no" \
        --target_label "$LABEL" \
        --target_label_id "$INDEX" \
        --model_path "/home/chenshengquan/program/fengsicheng/scBackdoor/test/scBERT-scBackdoor/checkpoint/TS_Heart_best.pth" \
        --dataset "TS_Heart" \
        --topnstop "$TOPN_STOP" >> "$OUTPUT_FILE3"

      torchrun --nproc_per_node=1 predict.py \
        --gene_num "3000" \
        --poisoned "yes" \
        --target_label "$LABEL" \
        --target_label_id "$INDEX" \
        --model_path "/home/chenshengquan/program/fengsicheng/scBackdoor/test/scBERT-scBackdoor/checkpoint/TS_Heart_best.pth" \
        --dataset "TS_Heart" \
        --topnstop "$TOPN_STOP" >> "$OUTPUT_FILE3"

      # End time and duration
      END_TIME=$(date +%s)
      DURATION=$((END_TIME - START_TIME))
      echo "Execution time : $DURATION seconds"
    done
  done
done


echo "Finish!---by Sicheng Feng"
