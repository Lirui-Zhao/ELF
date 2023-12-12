# [Boosting the Cross-Architecture Generalization of Dataset Distillation through an Empirical Study  (ELF)](https://arxiv.org/abs/2312.05598)

### Getting Started

First, download our repo:

```bash
git clone https://github.com/paper-code-anonymous/ELF.git
cd ELF
```

For an express instillation, we include ```.yaml``` files that you can run

```bash
conda env create -f requirements.yaml
```

You can then activate your  conda environment with

```bash
conda activate ELF
```

### Generate synthetic dataset

Below are some example commands to run each method, you can run one of them to generate synthetic dataset for eval using ELF.

#### Distillation with Distribution Matching

The following command will generate synthetic dataset using DM method.

```bash
sh distill_DM.sh
```

#### Distillation with Differentiable Siamese Augmentation

The following command will generate synthetic dataset using DSA method.

```bash
sh distill_DSA.sh
```

#### Distillation with Matching Training Trajectories

The following command will generate buffer for MTT method to distillation.

```bash
sh buffer_MTT.sh
```

The following command will generate synthetic dataset using MTT method.

```bash
sh distill_MTT.sh
```

### Eval using ELF

The following command will generate buffer for ELF to get features.

```bash
sh ELF_buffer.sh
```

The following command will eval synthetic dataset using ELF method.

```bash
sh ELF_eval.sh
```
