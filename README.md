<div align="center">
  <h3><b> Unveiling potential threats: backdoor attacks in single-cell pretrained models </b></h3>
</div>

<div align="center">

![](https://img.shields.io/github/last-commit/fscdc/scBackdoor?color=green)
![](https://img.shields.io/badge/PRs-Welcome-green)

</div>

<div align="center">


</div>

<p align="center">



</p>

---
>
> 🙋 Please let us know if you find out a mistake or have any suggestions!
> 
> 🌟 If you find this resource helpful, please consider to star this repository and cite our research:

```bibtex
Sicheng Feng, Siyu Li, Luonan Chen, Shengquan Chen. Unveiling potential threats: backdoor attacks in single-cell pretrained models. 2024.
```

<p align="center">
<img src="./figures/fig1.png" alt="" align=center style="width: 100%;" />
</p>

## Requirements and Installation
Use python 3.9 from Anaconda

- torch==2.1.2
- anndata==0.10.7
- datasets==2.19.1
- einops==0.8.0
- matplotlib==3.9.0
- numba==0.59.1                   
- numpy==1.26.3                   
- pandas==2.2.2                
- scanpy==1.10.1             
- scgpt==0.2.1                
- scikit-learn==1.5.0
- scipy==1.13.0 
- torchtext==0.16.2

To install all dependencies:

```bash
conda create -n scBackdoor python=3.9
conda activate scBackdoor
pip install -r requirements.txt
```

## Datasets
You can download the example datasets from [\[scGPT\]](https://github.com/bowang-lab/scGPT/blob/main/data/README.md) and [\[GeneFormer\]](https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset) , then place the downloaded contents under `Yourpath4Dataset` to reproduce the experiments.

## Pretrained Models
You can download the pretrained models from [\[scGPT\]](https://github.com/bowang-lab/scGPT/blob/main/README.md) (whole-human), [\[scBERT\]](https://github.com/TencentAILabHealthcare/scBERT) and [\[GeneFormer\]](https://huggingface.co/ctheodoris/Geneformer/tree/main), then place the downloaded contents under `Yourpath4PretrainedModels` to reproduce the experiments.

## Quick Demos
1. Download datasets and pretrained models, then place them under `rightpath` and adjust the path-params in the scripts.
2. Then you can try to reproduce the experiments with the provided scripts. For example, you can evaluate on *Human Pancreas* datasets by:

```bash
nohup ./run.sh & # for scGPT_Exp
```


## Details of Experiments

The commands to run the experiments are as follows:
```bash
nohup ./run.sh & # for scGPT_Exp
nohup ./run.sh & # for scBERT_Exp
python geneformer_scBackdoor.py # for GeneFormer_Exp
```

The poison-related code is in the `poison_utils.py` or `poison_trigger.py`. You can find them in each experiment's folder.

The folder tree is as follows:

```
├── LICENSE
├── README.md                             -- introduction about the project
├── figures                               -- use for show up
│   └── fig1.png
├── requirements.txt                      -- requirements for installation
│── scGPT_Exp                             
│   ├── test                              -- the attack pipeline
│   │   ├── run.sh
│   │   └── scBackdoor.py
│   └── utils                             -- the scGPT items
│       ├── detect_tools.py
│       ├── posion_trigger.py
│       ├── preprocess.py
│       ├── print_tools.py
│       └── tools.py
├── GeneFormer_Exp 
│   ├── geneformer                        -- the GeneFormer items
│   │   ├── __init__.py
│   │   ├── classifier.py
│   │   ├── classifier_utils.py
│   │   ├── collator_for_classification.py
│   │   ├── emb_extractor.py
│   │   ├── evaluation_utils.py
│   │   ├── gene_median_dictionary.pkl
│   │   ├── gene_name_id_dict.pkl
│   │   ├── in_silico_perturber.py
│   │   ├── in_silico_perturber_stats.py
│   │   ├── perturber_utils.py
│   │   ├── poison_utils.py
│   │   ├── pretrainer.py
│   │   ├── token_dictionary.pkl
│   │   └── tokenizer.py
│   └── geneformer_scBackdoor.py          -- the attack pipeline
└── scBERT_Exp
    ├── attn_sum_save.py
    ├── finetune.py
    ├── lr_baseline_crossorgan.py
    ├── performer_pytorch                 -- the scBERT items
    │   ├── __init__.py
    │   ├── performer_pytorch.py
    │   └── reversible.py
    ├── poison_utils.py
    ├── predict.py
    ├── preprocess.py
    ├── pretrain.py
    ├── run.sh                            -- the attack pipeline
    └── utils.py
```



## Further Reading
1. **scGPT: Towards Building a Foundation Model for Single-Cell Multi-omics Using Generative AI**, *Nature Methods* 2024.
[\[GitHub Repo\]](https://github.com/bowang-lab/scGPT)


2. **Transfer learning enables predictions in network biology**, *Nature* 2023.
[\[Huggingface Repo\]](https://huggingface.co/ctheodoris/Geneformer)


3. **scBERT as a Large-scale Pretrained Deep Language Model for Cell Type Annotation of Single-cell RNA-seq Data**, *Nature Machine Intelligence* 2022.
[\[GitHub Repo\]](https://github.com/TencentAILabHealthcare/scBERT)


## Acknowledgement

We sincerely thank the authors of the following open-source projects:

- [scGPT](https://github.com/bowang-lab/scGPT)
- [GeneFormer](https://huggingface.co/ctheodoris/Geneformer)
- [scBERT](https://github.com/TencentAILabHealthcare/scBERT)
- [scanpy](https://github.com/scverse/scanpy)
- [datasets](https://github.com/huggingface/datasets)
- [torch](https://pytorch.org/)