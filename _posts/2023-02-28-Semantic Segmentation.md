---
layout: post
title:  Semantic Segmentation of Thermal Facial Images
date:   2023-02-28 01:55:16
description: Real-world challenges in segmentation and proposed SAM-CL approach
tags: Adersarial-Learning, Contrastive-Learning, Semantic-Segmentation, Domain-Specific-Augmentation
categories: computer-vision
---

Segmentation of thermal facial images is a challenging task. This is because facial features often lack salience due to high-dynamic thermal range scenes and occlusion issues. Limited availability of datasets from unconstrained settings further limits the use of the state-of-the-art segmentation networks, loss functions and learning strategies which have been built and validated for RGB images. To address the challenge, we propose Self-Adversarial Multi-scale Contrastive Learning (SAM-CL) framework as a new training strategy for thermal image segmentation. SAM-CL framework consists of a SAM-CL loss function and a thermal image augmentation (TiAug) module as a domain-specific augmentation technique. We use the Thermal-Face-Database to demonstrate effectiveness of our approach. Experiments conducted on the existing segmentation networks (UNET, Attention-UNET, DeepLabV3 and HRNetv2) evidence the consistent performance gains from the SAM-CL framework. Furthermore, we present a qualitative analysis with UBComfort and DeepBreath datasets to discuss how our proposed methods perform in handling unconstrained situations.