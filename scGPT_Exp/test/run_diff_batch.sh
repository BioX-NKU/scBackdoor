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
LABELS_AND_INDEXES["CRABP1+ cells"]=0

TOPN_STOPS=("2.0")

# iterate over all combinations of parameters
for POISON_RATE in "${POISON_RATES[@]}"; do
  for LABEL in "${!LABELS_AND_INDEXES[@]}"; do
    INDEX=${LABELS_AND_INDEXES[$LABEL]}
    for TOPN_STOP in "${TOPN_STOPS[@]}"; do
      FORMATTED_LABEL=$(echo "$LABEL" | sed -e 's/ /_/g')

      # ouput file name
      OUTPUT_FILE1="${OUTPUT_DIR}/GSE261157-diff-batch.txt"
      OUTPUT_FILE2="${OUTPUT_DIR}/GSE261157-convert-diff-batch.txt"
      
      
      
      # Start time
      START_TIME=$(date +%s)

      # run the pipeline
      python -u scBackdoor.py \
        --dataset "GSE261157" \
        --poison_rate "$POISON_RATE" \
        --target_label "$LABEL" \
        --target_label_index "$INDEX" \
        --topn_stop "$TOPN_STOP" \
        --model_path "/home/chenshengquan/data/fengsicheng/scBackdoor/model/scGPT_human" \
        --poison "yes" > "$OUTPUT_FILE1"
        
      python -u scBackdoor.py \
        --dataset "GSE261157-convert" \
        --poison_rate "$POISON_RATE" \
        --target_label "$LABEL" \
        --target_label_index "$INDEX" \
        --topn_stop "$TOPN_STOP" \
        --model_path "/home/chenshengquan/data/fengsicheng/scBackdoor/model/scGPT_human" \
        --poison "yes" > "$OUTPUT_FILE2"
        

      # End time and duration
      END_TIME=$(date +%s)
      DURATION=$((END_TIME - START_TIME))
      echo "Execution time : $DURATION seconds"
    done
  done
done


declare -A LABELS_AND_INDEXES2
LABELS_AND_INDEXES2["CD4 t cell"]=0
LABELS_AND_INDEXES2["CD8 t cell"]=1
LABELS_AND_INDEXES2["CD24 Neutrophil"]=2
LABELS_AND_INDEXES2["NAMPT neutrophil"]=3
LABELS_AND_INDEXES2["NK cell"]=4
LABELS_AND_INDEXES2["erythrocyte"]=5
LABELS_AND_INDEXES2["erythroid progenitor"]=6
LABELS_AND_INDEXES2["granulocyte"]=7
LABELS_AND_INDEXES2["hematopoietic stem cell"]=8
LABELS_AND_INDEXES2["macrophage"]=9
LABELS_AND_INDEXES2["memory b cell"]=10
LABELS_AND_INDEXES2["monocyte"]=11
LABELS_AND_INDEXES2["myeloid progenitor"]=12
LABELS_AND_INDEXES2["naive b cell"]=13
LABELS_AND_INDEXES2["neutrophil"]=14
LABELS_AND_INDEXES2["plasma cell"]=15
LABELS_AND_INDEXES2["plasmablast"]=16

# iterate over all combinations of parameters
for POISON_RATE in "${POISON_RATES[@]}"; do
  for LABEL in "${!LABELS_AND_INDEXES2[@]}"; do
    INDEX=${LABELS_AND_INDEXES2[$LABEL]}
    for TOPN_STOP in "${TOPN_STOPS[@]}"; do
      FORMATTED_LABEL=$(echo "$LABEL" | sed -e 's/ /_/g')

      # ouput file name
      OUTPUT_FILE3="${OUTPUT_DIR}/TS-Bone-Marrow-diff-batch-$LABEL.txt"
      OUTPUT_FILE4="${OUTPUT_DIR}/TS-Bone-Marrow-convert-diff-batch-$LABEL.txt"
      
      # Start time
      START_TIME=$(date +%s)

      # run the pipeline        
      python -u scBackdoor.py \
        --dataset "TS-Bone-Marrow" \
        --poison_rate "$POISON_RATE" \
        --target_label "$LABEL" \
        --target_label_index "$INDEX" \
        --topn_stop "$TOPN_STOP" \
        --model_path "/home/chenshengquan/data/fengsicheng/scBackdoor/model/scGPT_human" \
        --poison "yes" > "$OUTPUT_FILE3"

      python -u scBackdoor.py \
        --dataset "TS-Bone-Marrow-convert" \
        --poison_rate "$POISON_RATE" \
        --target_label "$LABEL" \
        --target_label_index "$INDEX" \
        --topn_stop "$TOPN_STOP" \
        --model_path "/home/chenshengquan/data/fengsicheng/scBackdoor/model/scGPT_human" \
        --poison "yes" > "$OUTPUT_FILE4"

      # End time and duration
      END_TIME=$(date +%s)
      DURATION=$((END_TIME - START_TIME))
      echo "Execution time : $DURATION seconds"
    done
  done
done


declare -A LABELS_AND_INDEXES3
LABELS_AND_INDEXES3["Immune_notvalidated"]=0
LABELS_AND_INDEXES3["Schwann_cell"]=1
LABELS_AND_INDEXES3["basal_cells_confirmed"]=2
LABELS_AND_INDEXES3["capillary_endothelial_confirmed"]=3
LABELS_AND_INDEXES3["endothelial_cell_of_artery_confirmed"]=4
LABELS_AND_INDEXES3["endothelial_cell_of_lymphatic_vessel_confirmed"]=5
LABELS_AND_INDEXES3["epithelium_of_tongue_lowquality_TSP7_Anterior"]=6
LABELS_AND_INDEXES3["fibroblast"]=7
LABELS_AND_INDEXES3["pericyte_cell_confirmed"]=8
LABELS_AND_INDEXES3["tongue_keratinized_epithelium"]=9
LABELS_AND_INDEXES3["tongue_muscle_cell"]=10
LABELS_AND_INDEXES3["vein_endothelial_cell"]=11

# iterate over all combinations of parameters
for POISON_RATE in "${POISON_RATES[@]}"; do
  for LABEL in "${!LABELS_AND_INDEXES3[@]}"; do
    INDEX=${LABELS_AND_INDEXES3[$LABEL]}
    for TOPN_STOP in "${TOPN_STOPS[@]}"; do
      FORMATTED_LABEL=$(echo "$LABEL" | sed -e 's/ /_/g')

      # ouput file name
      OUTPUT_FILE5="${OUTPUT_DIR}/TS-Tongue-diff-batch-$LABEL.txt"
      OUTPUT_FILE6="${OUTPUT_DIR}/TS-Tongue-convert-diff-batch-$LABEL.txt"
      
      # Start time
      START_TIME=$(date +%s)

      # run the pipeline        
      python -u scBackdoor.py \
        --dataset "TS-Tongue" \
        --poison_rate "$POISON_RATE" \
        --target_label "$LABEL" \
        --target_label_index "$INDEX" \
        --topn_stop "$TOPN_STOP" \
        --model_path "/home/chenshengquan/data/fengsicheng/scBackdoor/model/scGPT_human" \
        --poison "yes" > "$OUTPUT_FILE5"

      python -u scBackdoor.py \
        --dataset "TS-Tongue-convert" \
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
