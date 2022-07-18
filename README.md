#  Few-Shot-Learning-Skin-Analysis

Model building, experiments, references and source code for the research work on skin image analysis using few-shot learning 

## Summary and Links to Papers

### *Baseline Reference* - [[CVPR-2020] Meta-DermDiagnosis Few-Shot Skin Disease Identification using Meta-Learning.pdf](./Literature/%5BCVPR-2020%5D%20Meta-DermDiagnosis%20Few-Shot%20Skin%20Disease%20Identification%20using%20Meta-Learning.pdf)

- Proposes the use of meta-learning techniques for efficient model adaptation for extremely low-data scenarios
- Applies Group equivariant convolutions (G-convolutions) in place of the normal spatial convolution filters
- Two network implementations: 
    - Reptile: Gradient-based meta-learning
    - Prototypical networks using Euclidean Distance
- Evaluated on ISIC 2018, Derm7pt and SD-198 datasets
- Outperforms DAML on ISIC 2018
- Implementation Code NOT available

### [[CVPR-2018] Learning to Compare Relation Network for Few-Shot Learning](./Literature/%5BCVPR-2018%5D%20Learning%20to%20Compare%20Relation%20Network%20for%20Few-Shot%20Learning.pdf)

- The paper that proposed Relation Networks for Few-Shot Learning

### [[NeurIPS-2017] Prototypical Networks for Few-shot Learning](./Literature/%5BNeurIPS-2017%5D%20Prototypical%20Networks%20for%20Few-shot%20Learning.pdf)

- The paper that proposed Protoypical Networks for Few-Shot Learning

### [[Elsevier-PR-2020] Temperature network for few-shot learning with distribution-aware large-margin metric](./Literature/%5BElsevier-PR-2020%5D%20Temperature%20network%20for%20few-shot%20learning%20with%20distribution-aware.pdf)

- An improvement of Prototypical Networks, by generating query-specific prototypes and thus results in local
and distribution-aware metric 
- Sets different temperature for different categories to penalize query samples that are not close enough to their belonging categories.
- *Code available - PyTorch*
