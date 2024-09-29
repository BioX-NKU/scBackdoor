import datetime
from geneformer.classifier import Classifier
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import warnings

warnings.filterwarnings("ignore")

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset", type=str, default="disease_state", help="Dataset name."
)

args = parser.parse_args()


output_prefix = args.dataset
output_dir = f"/home/chenshengquan/program/fengsicheng/scBackdoor/test/record-rebuttal/geneformer"

training_args = {
    "num_train_epochs": 0.9,
    "learning_rate": 0.000804,
    "lr_scheduler_type": "polynomial",
    "warmup_steps": 1812,
    "weight_decay": 0.258828,
    "per_device_train_batch_size": 12,
    "seed": 2024,
}

if args.dataset == "disease_state":
    filter_data_dict = {
        "cell_type": ["Cardiomyocyte1", "Cardiomyocyte2", "Cardiomyocyte3"]
    }
    # can use celltype to replace disease (label column)
    cell_state_dict = {"state_key": "disease", "states": "all"}

elif args.dataset == "celltype-liver":
    filter_data_dict = {"organ_major": ["liver"]}
    cell_state_dict = {"state_key": "cell_type", "states": "all"}
    
elif args.dataset == "celltype-pancreas":
    filter_data_dict = {"organ_major": ["pancreas"]}
    cell_state_dict = {"state_key": "cell_type", "states": "all"}
elif args.dataset == "celltype-bone_marrow":
    filter_data_dict = {"organ_major": ["bone_marrow"]}
    cell_state_dict = {"state_key": "cell_type", "states": "all"}
    
elif args.dataset == "celltype-lung":
    filter_data_dict = {"organ_major": ["lung"]}
    cell_state_dict = {"state_key": "cell_type", "states": "all"}
elif args.dataset == "celltype-spleen":
    filter_data_dict = {"organ_major": ["spleen"]}
    cell_state_dict = {"state_key": "cell_type", "states": "all"}
elif args.dataset == "celltype-kidney":
    filter_data_dict = {"organ_major": ["kidney"]}
    cell_state_dict = {"state_key": "cell_type", "states": "all"}
elif args.dataset == "celltype-immune":
    filter_data_dict = {"organ_major": ["immune"]}
    cell_state_dict = {"state_key": "cell_type", "states": "all"}
    
elif args.dataset == "celltype-large_intestine":
    filter_data_dict = {"organ_major": ["large_intestine"]}
    cell_state_dict = {"state_key": "cell_type", "states": "all"}
elif args.dataset == "celltype-placenta":
    filter_data_dict = {"organ_major": ["placenta"]}
    cell_state_dict = {"state_key": "cell_type", "states": "all"}
elif args.dataset == "celltype-brain":
    filter_data_dict = {"organ_major": ["brain"]}
    cell_state_dict = {"state_key": "cell_type", "states": "all"}
else:
    raise ValueError("Invalid dataset name: {}".format(args.dataset))


cc = Classifier(
    classifier="cell",
    cell_state_dict=cell_state_dict,
    filter_data=filter_data_dict,
    training_args=training_args,
    max_ncells=None,
    freeze_layers=2,
    num_crossval_splits=1,
    forward_batch_size=200,
    nproc=16,
    ngpu=4,
)

# Reference:
# https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main
# /example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset
if args.dataset == "disease_state":
    input_data_file = "/home/chenshengquan/data/fengsicheng/scBackdoor/data/human_dcm_hcm_nf_2048_w_length.dataset"

    # argument attr_to_split set to "individual" and attr_to_balance set to ["disease","lvef","age","sex","length"]
    train_ids = [
        "1447",
        "1600",
        "1462",
        "1558",
        "1300",
        "1508",
        "1358",
        "1678",
        "1561",
        "1304",
        "1610",
        "1430",
        "1472",
        "1707",
        "1726",
        "1504",
        "1425",
        "1617",
        "1631",
        "1735",
        "1582",
        "1722",
        "1622",
        "1630",
        "1290",
        "1479",
        "1371",
        "1549",
        "1515",
    ]
    eval_ids = ["1422", "1510", "1539", "1606", "1702"]
    test_ids = ["1437", "1516", "1602", "1685", "1718"]

    train_test_id_split_dict = {
        "attr_key": "individual",
        "train": train_ids + eval_ids,
        "test": test_ids,
    }

    train_valid_id_split_dict = {
        "attr_key": "individual",
        "train": train_ids,
        "eval": eval_ids,
    }
elif (
    args.dataset == "celltype-liver"
    or args.dataset == "celltype-pancreas"
    or args.dataset == "celltype-bone_marrow"
    or args.dataset == "celltype-lung"
    or args.dataset == "celltype-spleen"
    or args.dataset == "celltype-kidney"
    or args.dataset == "celltype-immune"
    or args.dataset == "celltype-large_intestine"
    or args.dataset == "celltype-placenta"
    or args.dataset == "celltype-brain"
):
    input_data_file = "/home/chenshengquan/data/fengsicheng/scBackdoor/data/cell_type_train_data.dataset"

    train_test_id_split_dict = None

    train_valid_id_split_dict = None
else:
    raise ValueError("Invalid dataset name: {}".format(args.dataset))


cc.prepare_data(
    input_data_file=input_data_file,
    output_directory=output_dir,
    output_prefix=output_prefix,
    split_id_dict=train_test_id_split_dict,
)


# Model refer: https://huggingface.co/ctheodoris/Geneformer/blob/main/model.safetensors
# Notice: Do not use this param: n_hyperopt_trials=3 (because of ray lib wrong)
all_metrics = cc.validate(
    model_directory="/home/chenshengquan/data/fengsicheng/scBackdoor/model/geneformer/",
    prepared_input_data_file=f"{output_dir}/{output_prefix}_labeled_train.dataset",
    id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
    output_directory=output_dir,
    output_prefix=output_prefix,
    split_id_dict=train_valid_id_split_dict,
)


cc = Classifier(
    classifier="cell",
    cell_state_dict=cell_state_dict,
    filter_data=filter_data_dict,
    forward_batch_size=150,
    nproc=16,
    ngpu=1,
)


# test clean data
all_metrics_test = cc.evaluate_saved_model(
    model_directory=f"{output_dir}/geneformer_Classifier_{output_prefix}/ksplit1/",
    id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
    test_data_file=f"{output_dir}/{output_prefix}_labeled_test.dataset",
    output_directory=output_dir,
    output_prefix=output_prefix,
)

print("Acc:", all_metrics_test["acc"])
print("Kappa:", all_metrics_test["kappa"])
print("Micro F1:", all_metrics_test["macro_f1"])

# test poisoned data
all_metrics_poisoned_test = cc.evaluate_saved_model(
    model_directory=f"{output_dir}/geneformer_Classifier_{output_prefix}/ksplit1/",
    id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
    test_data_file=f"{output_dir}/{output_prefix}_poisoned_labeled_test.dataset",
    output_directory=output_dir,
    output_prefix=output_prefix,
)

print("ASR:", all_metrics_poisoned_test["asr"])
