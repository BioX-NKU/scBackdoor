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

export CUDA_VISIBLE_DEVICES="3"

echo "CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"

# define parameters
POISON_RATES=("0.05")

declare -A LABELS_AND_INDEXES
# LABELS_AND_INDEXES["KIDNEY"]=1
# LABELS_AND_INDEXES["UCEC"]=7
LABELS_AND_INDEXES["cDC2"]=8
# LABELS_AND_INDEXES["THCA"]=6
# LABELS_AND_INDEXES["PAAD"]=5
# LABELS_AND_INDEXES["LYM"]=2

TOPN_STOPS=("2.0")

# iterate over all combinations of parameters
for POISON_RATE in "${POISON_RATES[@]}"; do
  for LABEL in "${!LABELS_AND_INDEXES[@]}"; do
    INDEX=${LABELS_AND_INDEXES[$LABEL]}
    for TOPN_STOP in "${TOPN_STOPS[@]}"; do
      FORMATTED_LABEL=$(echo "$LABEL" | sed -e 's/ /_/g')

      # ouput file name
      OUTPUT_FILE1="${OUTPUT_DIR}/3D-scgpt-mye-baseline.txt"

      # Start time
      START_TIME=$(date +%s)

      # run the pipeline
      python -u scBackdoor.py \
        --dataset "mye" \
        --poison_rate "$POISON_RATE" \
        --target_label "$LABEL" \
        --target_label_index "$INDEX" \
        --topn_stop "$TOPN_STOP" \
        --model_path "/home/chenshengquan/data/fengsicheng/scBackdoor/model/scGPT_human" \
        --poison "no" > "$OUTPUT_FILE1"
        
        
      # End time and duration
      END_TIME=$(date +%s)
      DURATION=$((END_TIME - START_TIME))
      echo "Execution time : $DURATION seconds"
    done
  done
done


declare -A LABELS_AND_INDEXES2
LABELS_AND_INDEXES2["Cardiac Fibroblast"]=0
# LABELS_AND_INDEXES2["Cardiac Muscle Cells"]=1
# LABELS_AND_INDEXES2["Macrophages"]=4
# LABELS_AND_INDEXES2["Smooth Muscle Cells"]=5
# use rare type
# {0: 'Cardiac Fibroblast', 1: 'Cardiac Muscle Cells', 2: 'Endothelial Cells', 3: 'Hepatocyte', 4: 'Macrophages', 5: 'Smooth Muscle Cells'}

# iterate over all combinations of parameters
for POISON_RATE in "${POISON_RATES[@]}"; do
  for LABEL in "${!LABELS_AND_INDEXES2[@]}"; do
    INDEX=${LABELS_AND_INDEXES2[$LABEL]}
    for TOPN_STOP in "${TOPN_STOPS[@]}"; do
      FORMATTED_LABEL=$(echo "$LABEL" | sed -e 's/ /_/g')

      # ouput file name
      OUTPUT_FILE2="${OUTPUT_DIR}/3D-scgpt-TS-Heart-baseline.txt"

      # Start time
      START_TIME=$(date +%s)

      # run the pipeline
      python -u scBackdoor.py \
        --dataset "TS_Heart" \
        --poison_rate "$POISON_RATE" \
        --target_label "$LABEL" \
        --target_label_index "$INDEX" \
        --topn_stop "$TOPN_STOP" \
        --model_path "/home/chenshengquan/data/fengsicheng/scBackdoor/model/scGPT_human" \
        --poison "no" > "$OUTPUT_FILE2"
        
        
      # End time and duration
      END_TIME=$(date +%s)
      DURATION=$((END_TIME - START_TIME))
      echo "Execution time : $DURATION seconds"
    done
  done
done


declare -A LABELS_AND_INDEXES3
LABELS_AND_INDEXES3["B"]=0
# LABELS_AND_INDEXES3["CD4+ T"]=1
# LABELS_AND_INDEXES3["CD8+ T"]=2
# LABELS_AND_INDEXES3["Endothelial"]=3
# LABELS_AND_INDEXES3["Epithelial"]=4
# LABELS_AND_INDEXES3["Fibroblast"]=5
# LABELS_AND_INDEXES3["Glial"]=6
# LABELS_AND_INDEXES3["Innate lymphoid"]=7
# LABELS_AND_INDEXES3["Mast"]=8
# LABELS_AND_INDEXES3["Mural"]=9
# LABELS_AND_INDEXES3["Myeloid"]=10
# LABELS_AND_INDEXES3["Plasma"]=11

# iterate over all combinations of parameters
for POISON_RATE in "${POISON_RATES[@]}"; do
  for LABEL in "${!LABELS_AND_INDEXES3[@]}"; do
    INDEX=${LABELS_AND_INDEXES3[$LABEL]}
    for TOPN_STOP in "${TOPN_STOPS[@]}"; do
      FORMATTED_LABEL=$(echo "$LABEL" | sed -e 's/ /_/g')

      # ouput file name
      OUTPUT_FILE3="${OUTPUT_DIR}/3D-scgpt-GSE206785-baseline.txt"

      # Start time
      START_TIME=$(date +%s)

      # run the pipeline
      python -u scBackdoor.py \
        --dataset "GSE206785" \
        --poison_rate "$POISON_RATE" \
        --target_label "$LABEL" \
        --target_label_index "$INDEX" \
        --topn_stop "$TOPN_STOP" \
        --model_path "/home/chenshengquan/data/fengsicheng/scBackdoor/model/scGPT_human" \
        --poison "no" > "$OUTPUT_FILE3"
        
        
      # End time and duration
      END_TIME=$(date +%s)
      DURATION=$((END_TIME - START_TIME))
      echo "Execution time : $DURATION seconds"
    done
  done
done


echo "Finish!---by Sicheng Feng"
