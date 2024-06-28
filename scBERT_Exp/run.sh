#!/bin/bash


OUTPUT_DIR="/home/chenshengquan/data/fengsicheng/scBackdoor/records2/Pancreas-scBERT/"
mkdir -p "${OUTPUT_DIR}"


POISON_RATES=("0.05")


declare -A LABELS_AND_INDEXES
LABELS_AND_INDEXES["PP"]=0
# LABELS_AND_INDEXES["PSC"]=1
# LABELS_AND_INDEXES["acinar"]=2
# LABELS_AND_INDEXES["alpha"]=3
# LABELS_AND_INDEXES["beta"]=4
# LABELS_AND_INDEXES["delta"]=5
# LABELS_AND_INDEXES["ductal"]=6
# LABELS_AND_INDEXES["endothelial"]=7
# LABELS_AND_INDEXES["epsilon"]=8
# LABELS_AND_INDEXES["macrophage"]=9
# LABELS_AND_INDEXES["mast"]=10
# LABELS_AND_INDEXES["schwann"]=11
# LABELS_AND_INDEXES["t_cell"]=12


TOPN_STOPS=("2.0")

# iterate over all combinations of parameters
for POISON_RATE in "${POISON_RATES[@]}"; do
  for LABEL in "${!LABELS_AND_INDEXES[@]}"; do
    INDEX=${LABELS_AND_INDEXES[$LABEL]}
    for TOPN_STOP in "${TOPN_STOPS[@]}"; do
      FORMATTED_LABEL=$(echo "$LABEL" | sed -e 's/ /_/g')

      # ouput file name
      OUTPUT_FILE="${OUTPUT_DIR}/${POISON_RATE}-${INDEX}-${TOPN_STOP}.txt"
#       OUTPUT_FILE="${OUTPUT_DIR}/baseline.txt"

      # Start time
      START_TIME=$(date +%s)

      # run the pipeline
      torchrun --nproc_per_node=1 finetune.py \
        --gene_num "3000" \
        --poisoned "yes" \
        --poison_rate "$POISON_RATE" \
        --target_label "$LABEL" \
        --target_label_id "$INDEX" \
        --topnstop "$TOPN_STOP" > "$OUTPUT_FILE"

      python predict.py \
        --gene_num "3000" \
        --poisoned "no" \
        --target_label "$LABEL" \
        --target_label_id "$INDEX" \
        --topnstop "$TOPN_STOP" >> "$OUTPUT_FILE"

      python predict.py \
        --gene_num "3000" \
        --poisoned "yes" \
        --target_label "$LABEL" \
        --target_label_id "$INDEX" \
        --topnstop "$TOPN_STOP" >> "$OUTPUT_FILE"

      # End time and duration
      END_TIME=$(date +%s)
      DURATION=$((END_TIME - START_TIME))
      echo "Execution time : $DURATION seconds"
    done
  done
done


echo "Finished Running."
