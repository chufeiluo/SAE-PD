# SAE-PD

This is the main directory for the SAE-PD project, which attempts to capture the effect of pluralistic human feedback through steering vectors and contrastive decoding instead of model training.

Human feedback can be dense and diverse for many subjects such as hate speech. We test how well we can model the prior of subjective tasks over a large batch of human opinions **without model training.** We accomplish this by using steering vectors to capture unique perspectives instead of model training - theoretically, if a human's perspective is sufficiently consistent, we explore whether or not this can translate into a consistent displacement in the LLM latent space.

## How to run
1. Install dependencies using requirements.txt
2. Modify the dataset directories in the appropriate loading function (load_gqa, load_hs, load_mislc)
2. Run the script run_experiments_clean.sh, either locally or through slurm

## Citing this work
We have a short paper published in EMNLP 2025! If you found this repository helpful, please cite our work:
```
@inproceedings{luo-etal-2025-towards,
    title = "Towards Low-Resource Alignment to Diverse Perspectives with Sparse Feedback",
    author = "Luo, Chu Fei  and
      Dahan, Samuel  and
      Zhu, Xiaodan",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2025",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-emnlp.1106/",
    doi = "10.18653/v1/2025.findings-emnlp.1106",
}
```

