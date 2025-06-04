---
layout: post
title: Matrix Factorization as Attention - Rethinking Multidimensional Feature Processing in Remote Physiological Sensing
date: 2025-06-02 16:07:30
description: Deciphering matrix factorization-based attention mechanisms and how they differ from cross-attention/transformers.
tags: matrix-factorization, attention, deep-learning, computer-vision, physiological-sensing
categories: computer-vision
author: Jitesh Joshi, Youngjun Cho
---

<!-- ## How can factorization serve as attention? -->

We presented our work - FactorizePhys [[1]](#references), that focuses on remote photoplethysmography (rPPG), at NeurIPS 2024, held at Vancouver, Canada during 10th to 14th December. The conference was extremely memorable, and our work was appreciated by several attendees who visited our poster session. Some obvious queries from fellow researchers stressed why not use transformer networks, when cross-attention has been the backbone of LLM advancements. Researchers even attempted to draw parallels with cross-attention formulation to understand our proposed matrix factorization-based multidimensional attention (FSAM), with their key concern being: **how can factorization serve as attention?**

Here, we explore the rationale behind matrix factorization-based attention mechanisms and how they differ from self-attention/ cross-attention/ transformers.

<!-- ![FactorizePhys Poster](/assets/img/factorizephys/FactorizePhys_Poster.png) -->
<div align="center">
    <img src="/assets/img/factorizephys/FactorizePhys_Poster.png" alt="FactorizePhys Poster" style="width:95%; height:auto;">
</div>

*Figure 1: Our poster at NeurIPS 2024, where we discussed FSAM with fellow researchers from the computer vision and machine learning community.*


## The Compression-as-Attention Paradigm

Using compression as an attention mechanism isn't new. In the CNN-dominated era, squeeze-and-excitation (SE) attention [[2]](#references) was among the most popular mechanisms. SE attention works by **globally average pooling** across spatial dimensions to compress features into channel descriptors, then using fully connected layers to model channel interdependencies, and finally rescaling the original features.

However, a fundamental limitation emerges when working with multidimensional feature spaces: **existing attention mechanisms compute attention disjointly across spatial, temporal, and channel dimensions**. For tasks like rPPG estimation that require joint modeling of these dimensions, squeezing individual dimensions can result in information loss, causing learned attention to miss comprehensive multidimensional feature relationships.

This is precisely the problem our work addresses.

## FSAM: Factorized Self-Attention Module

FSAM uses Non-negative Matrix Factorization (NMF) [[3]](#references) to factorize multidimensional feature space into a low-rank approximation, serving as a compressed representation that preserves interdependencies across all dimensions. The key advantages are:

1. **Joint multidimensional attention** - No dimension squeezing required; processes spatial, temporal, and channel dimensions simultaneously
2. **Parameter-free optimization** - Uses classic NMF as proposed by Lee & Seung, 1999 [[3]](#references), with an optimization algorithm approximated as multiplicative updates [[4]](#references), implemented under 'no_grad' block
3. **Task-specific design** - Tailored for signal extraction tasks with rank-1 factorization

<!-- ![FSAM Overview](/assets/img/factorizephys/FSAM.png) -->
<div align="center">
    <img src="/assets/img/factorizephys/FSAM.png" alt="FSAM Overview" style="width:60%; height:auto;">
</div>

*Figure 2: Overview of the Factorized Self-Attention Module (FSAM) showing how multidimensional voxel embeddings are transformed into a 2D matrix, factorized using NMF, and reconstructed to provide attention weights.*

## Mathematical Formulation

### The Critical Transformation

For input spatial-temporal data **$I ∈ ℝ^(T×C×H×W)$**, we generate voxel embeddings **$ω ∈ ℝ^(ω×ε×ϑ×ϖ)$** through 3D feature extraction. The **core innovation** lies in how we reshape these embeddings for factorization.

**Traditional 2D approach** (like Hamburger module [[4]](#references)):

$$V_s ∈ ℝ^(M×N) where: ϑ (channels) → M, ϖ×ϱ (spatial) → N$$

**Our 3D spatial-temporal approach**:

$$V_st ∈ ℝ^(M×N) where: ε (temporal) → M, ϑ×ϖ×ϱ (spatial+channel) → N$$

This transformation is **crucial** for rPPG estimation because:

- **Physiological signal correlation**: We need correlations between spatial/channel features and temporal patterns for BVP signal recovery
- **Single signal source**: Only one underlying BVP signal across facial regions justifies rank-1 factorization ($L=1$)
- **Scale considerations**: Temporal and spatial dimensions have vastly different scales (typically $ε >> ϖ, ϱ$ for video data)

### The NMF Attention Mechanism

The factorization process uses iterative multiplicative updates:

```python
# Preprocessing: ensure non-negativity for NMF
x = ReLU(Conv3D(ω - ω.min()))

# Transform to factorization matrix
V_st = reshape(x, (batch×S, temporal_dim, spatial×channel_dim))

# Initialize bases and coefficients
bases = ones(batch×S, temporal_dim, rank=1)
coef = softmax(V_st^T @ bases)

# Iterative multiplicative updates (4-8 steps)
for step in range(MD_STEPS):
    # Update coefficients
    numerator = V_st^T @ bases
    denominator = coef @ (bases^T @ bases)
    coef = coef ⊙ (numerator / (denominator + ε))
    
    # Update bases  
    numerator = V_st @ coef
    denominator = bases @ (coef^T @ coef)
    bases = bases ⊙ (numerator / (denominator + ε))

# Reconstruct attention
V̂_st = bases @ coef^T
ω̂ = reshape_back(V̂_st)

# Apply attention with residual connection
output = ω + InstanceNorm(ω ⊙ postprocess(ω̂))
```

<!-- ![NMF Algorithm Flowchart](images/nmf-algorithm-flowchart.png)
*Figure 5: Flowchart of the NMF-based attention mechanism showing the iterative multiplicative updates process and how the low-rank approximation is applied as attention.* -->

### Why Rank-1 Factorization Works

The paper's ablation studies confirm that **rank-1 factorization performs optimally** for rPPG estimation. This aligns with the physiological assumption that there's a single underlying blood volume pulse signal across different facial regions. Higher ranks (L > 1) showed performance comparable to the baseline without FSAM, indicating rank-1 captures the essential signal structure.

*Table 1: Ablation study results showing performance across different factorization ranks. Rank-1 achieves optimal performance, supporting the single signal source assumption for rPPG estimation.*

<!-- ![Rank Ablation Study](assets/img/factorizephys/rank-ablation.png) -->
<div align="center">
    <img src="/assets/img/factorizephys/rank-ablation.png" alt="Rank Ablation Study" style="width:70%; height:auto;">
</div>

## Why FSAM Outperforms Transformers

### 1. **Task-Specific vs Generic Design**

**Transformers** use generic self-attention that treats all positions equally:

$$Attention(Q,K,V) = softmax(QK^T/√d_k)V$$

**FSAM** is specifically designed for spatial-temporal signal extraction:

- **Temporal vectors as the primary dimension** (signals evolve over time)
- **Spatial-channel features as descriptors** (different facial regions contribute differently)
- **Rank-1 constraint** enforces single signal source assumption

### 2. **Computational Efficiency**

- **FSAM complexity**: O(n) with 4-8 multiplicative update steps
- **Transformer complexity**: O(n²) with full attention computation [[5]](#references)
- **Parameter comparison**: FactorizePhys (52K) vs PhysFormer (7.38M) - **138x fewer parameters**

<!-- ![Attention Mechanism Comparison](images/transformer-vs-fsam-attention.png)
*Figure 7: Conceptual comparison between Transformer self-attention (left) and FSAM factorization-based attention (right), highlighting the task-specific design of FSAM for signal extraction.* -->

### 3. **Superior Cross-Dataset Generalization**

**Table 2: Comprehensive evaluation across four datasets shows remarkable generalization**

| Training → Testing | PhysFormer (MAE↓) | EfficientPhys (MAE↓) | **FactorizePhys (MAE↓)** |
|:------------------ |:-----------------:|:--------------------:|:------------------------:|
| iBVP → PURE        | 6.58 ± 1.98       | 0.56 ± 0.17          | **0.60 ± 0.21**          |
| SCAMPS → PURE      | 16.64 ± 2.95      | 6.21 ± 2.26          | **5.43 ± 1.93**          |
| UBFC → PURE        | 8.90 ± 2.15       | 4.71 ± 1.79          | **0.48 ± 0.17**          |


**Key insight**: When trained on synthetic data (SCAMPS) and tested on real data, FactorizePhys shows the smallest performance gap, indicating superior domain transfer.

*Table 3: Cross-dataset generalization performance comparison*

<!-- ![Cross-Dataset Performance](/assets/img/factorizephys/cross-dataset-performance.png) -->
<div align="center">
    <img src="/assets/img/factorizephys/cross-dataset-performance.png" alt="Cross-Dataset Performance" style="width:80%; height:auto;">
</div>

FactorizePhys consistently outperforms existing state-of-the-art methods, including the transformer-based methods across different domain shifts, particularly in synthetic-to-real transfer scenarios.

### 4. **Attention Visualization**

Our cosine similarity visualization between temporal embeddings and ground-truth PPG signals reveals:

- **Higher correlation scores** for FactorizePhys with FSAM
- **Better spatial selectivity** - correctly identifies skin regions with strong pulse signals

<!-- ![Attention Visualization](assets/img/factorizephys/Attention_Maps.png) -->
<div align="center">
    <img src="/assets/img/factorizephys/Attention_Maps.png" alt="Attention Visualization" style="width:70%; height:auto;">
</div>

*Figure 3: Attention visualization comparing baseline model (left) and FactorizePhys with FSAM (right). Higher cosine similarity scores (brighter regions) indicate better spatial selectivity for pulse-rich facial regions.*

## The Inference-Time Advantage

A surprising finding: **FactorizePhys trained with FSAM retains performance even when FSAM is dropped during inference**. This suggests FSAM enhances saliency of relevant features during training, guiding the network to learn such salient feature representations that persist without the attention module.

```python
# Training: FSAM influences 3D convolutional kernels
factorized_embeddings = fsam(voxel_embeddings)
loss = compute_loss(head(factorized_embeddings), ground_truth)

# Inference: Can drop FSAM without performance loss
output = head(voxel_embeddings)  # No FSAM needed!
```

This dramatically reduces inference latency while maintaining accuracy - ideal for real-time applications.

<!-- ![Inference Time Comparison](/assets/img/factorizephys/Inference-latency.png) -->
<div align="center">
    <img src="/assets/img/factorizephys/Inference-latency.png" alt="Inference Time Comparison" style="width:60%; height:auto;">
</div>

*Figure 4: Performance vs. latency comparison showing FactorizePhys achieves superior accuracy with minimal inference time, especially when FSAM is dropped during inference.*

## Key Contributions and Broader Implications

### For the rPPG Community

- **First matrix factorization-based attention** specifically designed for physiological signal extraction from facial videos
- **State-of-the-art cross-dataset generalization** with dramatically fewer parameters
- **Real-time deployment capability** without performance degradation

### For the Broader AI Community

- **Novel attention paradigm** demonstrating that domain-specific designs can outperform generic mechanisms
- **Efficiency breakthrough**: 138x parameter reduction compared to transformers with superior performance
- **New perspective on attention in deep-learing architecutures**: Factorization as compression can be more effective than dimension squeezing

<!-- ![Impact Summary](images/contributions-impact-summary.png)
*Figure 11: Summary of key contributions and their impact on both the rPPG research community and broader AI/computer vision fields.* -->

## Looking Forward: Future Research Directions

FSAM's success opens several promising avenues:

1. **Extended Applications**: Video understanding, action recognition, time-series analysis
2. **Enhanced NMF Variants**: Incorporating temporal smoothness or frequency domain constraints. Checkout our subsequent work ([MMRPhys](https://arxiv.org/abs/2505.07013)) that explores this direction for robust estimation of multiple physiological signals.
3. **Hybrid Architectures**: Combining factorization-based attention with other mechanisms for different modalities
4. **Theoretical Analysis**: Understanding why rank-1 factorization generalizes so well across datasets

<!-- 
![Future Directions](images/future-research-directions.png)
*Figure 12: Roadmap of future research directions enabled by the FSAM framework, spanning from theoretical analysis to practical applications across different domains.* -->

## Conclusion

The key insight is profound yet simple: **not all tasks require the full complexity of transformer attention**. For spatial-temporal signal extraction tasks, a well-designed, task-specific attention mechanism can achieve superior robustness with dramatically improved efficiency.

FSAM demonstrates that the deeper understanding the problem domain can lead to more effective solutions than applying generic, computationally expensive methods. In an era of ever-growing model sizes, this work shows that **thoughtful design trumps brute-force scaling**.

---

## Code and Data Availability

| Resources   |  | Link |
|-------------|--|------|
| **Paper**   |  | [FactorizePhys](https://proceedings.neurips.cc/paper_files/paper/2024/hash/af1c61e4dd59596f033d826419870602-Abstract-Conference.html) |
| **Code**    |  | [GitHub](https://github.com/PhysiologicAILab/FactorizePhys) |
| **Dataset** |  | [iBVP Dataset](https://github.com/PhysiologicAILab/iBVP-Dataset) |

---

## References

[1] Joshi, J., Agaian, S. S., & Cho, Y. (2024). FactorizePhys: Matrix Factorization for Multidimensional Attention in Remote Physiological Sensing. In *Advances in Neural Information Processing Systems* (NeurIPS 2024).

[2] Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-excitation networks. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 7132-7141).

[3] Lee, D. D., & Seung, H. S. (1999). Learning the parts of objects by non-negative matrix factorization. *Nature*, 401(6755), 788-791.

[4] Geng, Z., Guo, M. H., Chen, H., Li, X., Wei, K., & Lin, Z. (2021). Is attention better than matrix decomposition? In *International Conference on Learning Representations*.

[5] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In *Advances in Neural Information Processing Systems* (pp. 5998-6008).

[6] Yu, Z., Shen, Y., Shi, J., Zhao, H., Torr, P. H., & Zhao, G. (2022). PhysFormer: Facial video-based physiological measurement with temporal difference transformer. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 4186-4196).

[7] Liu, X., Hill, B., Jiang, Z., Patel, S., & McDuff, D. (2023). EfficientPhys: Enabling simple, fast and accurate camera-based cardiac measurement. In *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision* (pp. 5008-5017).

[8] Stricker, R., Müller, S., & Gross, H. M. (2014). Non-contact video-based pulse rate measurement on a mobile service robot. In *The 23rd IEEE International Symposium on Robot and Human Interactive Communication* (pp. 1056-1062).

[9] McDuff, D., Wander, M., Liu, X., Hill, B., Hernandez, J., Lester, J., & Baltrusaitis, T. (2022). SCAMPS: Synthetics for camera measurement of physiological signals. *Advances in Neural Information Processing Systems*, 35, 3744-3757.

[10] Bobbia, S., Macwan, R., Benezeth, Y., Mansouri, A., & Dubois, J. (2019). Unsupervised skin tissue segmentation for remote photoplethysmography. *Pattern Recognition Letters*, 124, 82-90.

---

*This work was conducted at the Department of Computer Science, University College London, under the supervision of Prof. Youngjun Cho. Jitesh Joshi was fully supported with international studentship that was secured by Prof. Cho*

*For questions or collaborations, please contact: [jitesh.joshi.20@ucl.ac.uk](mailto:jitesh.joshi.20@ucl.ac.uk)*