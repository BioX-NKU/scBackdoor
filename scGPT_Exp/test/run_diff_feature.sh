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

export CUDA_VISIBLE_DEVICES="0"

echo "CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"


# define parameters
POISON_RATES=("0.05")

declare -A LABELS_AND_INDEXES
LABELS_AND_INDEXES["PP"]=1
# LABELS_AND_INDEXES["PSC"]=2
# LABELS_AND_INDEXES["acinar"]=3
# LABELS_AND_INDEXES["alpha"]=4
# LABELS_AND_INDEXES["beta"]=5
# LABELS_AND_INDEXES["delta"]=6
# LABELS_AND_INDEXES["ductal"]=7
# LABELS_AND_INDEXES["endothelial"]=8
# LABELS_AND_INDEXES["epsilon"]=9
# LABELS_AND_INDEXES["macrophage"]=10
# LABELS_AND_INDEXES["mast"]=11
# LABELS_AND_INDEXES["schwann"]=12
# LABELS_AND_INDEXES["t_cell"]=13

TOPN_STOPS=("2.0")

# iterate over all combinations of parameters
for POISON_RATE in "${POISON_RATES[@]}"; do
  for LABEL in "${!LABELS_AND_INDEXES[@]}"; do
    INDEX=${LABELS_AND_INDEXES[$LABEL]}
    for TOPN_STOP in "${TOPN_STOPS[@]}"; do
      FORMATTED_LABEL=$(echo "$LABEL" | sed -e 's/ /_/g')

      # ouput file name
      OUTPUT_FILE1="${OUTPUT_DIR}/pancreas-diff-features-2000-2000.txt"
      OUTPUT_FILE2="${OUTPUT_DIR}/pancreas-diff-features-3000-2750.txt"
      OUTPUT_FILE3="${OUTPUT_DIR}/pancreas-diff-features-3000-2500.txt"
      OUTPUT_FILE4="${OUTPUT_DIR}/pancreas-diff-features-3000-2250.txt"
      OUTPUT_FILE5="${OUTPUT_DIR}/pancreas-diff-features-3000-2000.txt"
      OUTPUT_FILE6="${OUTPUT_DIR}/pancreas-diff-features-3000-300.txt"

      # Start time
      START_TIME=$(date +%s)

      # run the pipeline
      python -u scBackdoor.py \
        --dataset "pancreas-diff-features-2000-2000" \
        --poison_rate "$POISON_RATE" \
        --target_label "$LABEL" \
        --target_label_index "$INDEX" \
        --topn_stop "$TOPN_STOP" \
        --model_path "/home/chenshengquan/data/fengsicheng/scBackdoor/model/scGPT_human" \
        --poison "yes" > "$OUTPUT_FILE1"
        
        
      python -u scBackdoor.py \
        --dataset "pancreas-diff-features-3000-2750" \
        --poison_rate "$POISON_RATE" \
        --target_label "$LABEL" \
        --target_label_index "$INDEX" \
        --topn_stop "$TOPN_STOP" \
        --model_path "/home/chenshengquan/data/fengsicheng/scBackdoor/model/scGPT_human" \
        --poison "yes" > "$OUTPUT_FILE2"
        
    
      python -u scBackdoor.py \
        --dataset "pancreas-diff-features-3000-2500" \
        --poison_rate "$POISON_RATE" \
        --target_label "$LABEL" \
        --target_label_index "$INDEX" \
        --topn_stop "$TOPN_STOP" \
        --model_path "/home/chenshengquan/data/fengsicheng/scBackdoor/model/scGPT_human" \
        --poison "yes" > "$OUTPUT_FILE3"


      python -u scBackdoor.py \
        --dataset "pancreas-diff-features-3000-2250" \
        --poison_rate "$POISON_RATE" \
        --target_label "$LABEL" \
        --target_label_index "$INDEX" \
        --topn_stop "$TOPN_STOP" \
        --model_path "/home/chenshengquan/data/fengsicheng/scBackdoor/model/scGPT_human" \
        --poison "yes" > "$OUTPUT_FILE4"
        
        
      python -u scBackdoor.py \
        --dataset "pancreas-diff-features-3000-2000" \
        --poison_rate "$POISON_RATE" \
        --target_label "$LABEL" \
        --target_label_index "$INDEX" \
        --topn_stop "$TOPN_STOP" \
        --model_path "/home/chenshengquan/data/fengsicheng/scBackdoor/model/scGPT_human" \
        --poison "yes" > "$OUTPUT_FILE5"


      python -u scBackdoor.py \
        --dataset "pancreas-diff-features-3000-300" \
        --poison_rate "$POISON_RATE" \
        --target_label "$LABEL" \
        --target_label_index "$INDEX" \
        --topn_stop "$TOPN_STOP" \
        --model_path "/home/chenshengquan/data/fengsicheng/scBackdoor/model/scGPT_human" \
        --poison "yes" > "$OUTPUT_FILE6"

      # End time and duration
      END_TIME=$(date +%s)
      DURATION=$((END_TIME - START_TIME))
      echo "Execution time : $DURATION seconds"
    done
  done
done

echo "Finish!---by Sicheng Feng"
