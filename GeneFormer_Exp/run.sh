export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

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

export CUDA_VISIBLE_DEVICES="0,1,2,3"

echo "CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"

# original datasets
DATASETS=("disease_state")

for DATASET in "${DATASETS[@]}"; do

    OUTPUT_FILE1="${OUTPUT_DIR}/3D-geneformer-disease.txt"

    # Start time
    START_TIME=$(date +%s)

#     python scBackdoor_GeneFormer.py \
#     --dataset "$DATASET" > "$OUTPUT_FILE1"
    
    # End time and duration
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo "Execution time : $DURATION seconds"
done

# three new datasets
DATASETS2=("celltype-immune")

for DATASET in "${DATASETS2[@]}"; do

    OUTPUT_FILE2="${OUTPUT_DIR}/3D-geneformer-celltype-immune.txt"

    # Start time
    START_TIME=$(date +%s)

    python scBackdoor_GeneFormer.py \
    --dataset "$DATASET" > "$OUTPUT_FILE2"
    
    # End time and duration
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo "Execution time : $DURATION seconds"
done


DATASETS4=("celltype-spleen")

for DATASET in "${DATASETS4[@]}"; do

    OUTPUT_FILE4="${OUTPUT_DIR}/3D-geneformer-celltype-spleen.txt"

    # Start time
    START_TIME=$(date +%s)

    python scBackdoor_GeneFormer.py \
    --dataset "$DATASET" > "$OUTPUT_FILE4"
    
    # End time and duration
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo "Execution time : $DURATION seconds"
done


DATASETS5=("celltype-brain")

for DATASET in "${DATASETS5[@]}"; do

    OUTPUT_FILE4="${OUTPUT_DIR}/3D-geneformer-celltype-brain.txt"

    # Start time
    START_TIME=$(date +%s)

    python scBackdoor_GeneFormer.py \
    --dataset "$DATASET" > "$OUTPUT_FILE4"
    
    # End time and duration
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo "Execution time : $DURATION seconds"
done
