# Customizing Visual Emotion Evaluation for MLLMs  

[![Paper](https://img.shields.io/badge/Paper-arXiv-blue)](link-to-paper)  [![Data](https://img.shields.io/badge/Data-HuggingFace-orange)](link-to-benchmark)  

Project page of:  
**Customizing Visual Emotion Evaluation for MLLMs: An Open-Vocabulary, Multifaceted, and Scalable Approach**  
*Daiqing Wu, Dongbao Yang, Sicheng Zhao, Can Ma, Yu Zhou*  

---

# TODO List

## ✅ Completed

- [x] Create repository and setup project overview
- [x] Upload MVEI benchmark dataset
  - [x] Include open-vocabulary emotion annotations
  - [x] Include emotion-centric statements
  - [x] Add dataset documentation and usage examples

## 🔄 In Progress

- [ ] Release standard evaluation script
- [ ] Upload INSETS-462k corpus
  - [ ] Include open-vocabulary emotion annotations
  - [ ] Include emotion-centric statements
  - [ ] Include intermediate results, i.e., candidate emotions, prototype statements
- [ ] Release INSETS pipeline code

---

## 📖 Overview  

This repository provides the **code and data** introduced in our paper.  
We propose a comprehensive framework for evaluating the visual emotion intelligence of **Multimodal Large Language Models (MLLMs)**, consisting of four key components:

- **ESJ Task**: A judgment-based evaluation task to assess MLLMs' emotion perception.
- **INSETS Pipeline**: An automated pipeline for open-vocabulary labels and multi-faceted emotion statements.  
- **INSETS-462k Dataset**: A large-scale automatically annotated corpus for ESJ.
- **MVEI**: A human-refined benchmark for multifaceted visual emotion intelligence evaluation.

---

## ESJ Task

The **Emotion Statement Judgment (ESJ)** task reformulates visual emotion evaluation by requiring MLLMs to validate whether a given emotion-centric statement accurately describes the emotional content of an image. This approach mitigates the ambiguity of open-ended responses and supports extensible evaluation across multiple affective dimensions.

![ESJ Task](images/ESJ.jpeg)
<div align="center">
<em>Figure 1: Comparison between traditional emotion evaluation approaches and the proposed ESJ task.</em>
</div>

---

## INSETS Pipeline

The **INSETS** pipeline (**I**ntelligent **Vi**sual **E**motion **T**agger and **S**tatement **C**onstructor) automatically constructs emotion-centric statements for ESJ with minimal human effort. It operates in two main stages:

**Open-Vocabulary Emotion Tagging**: Assigns fine-grained emotion labels to images using an ensemble of MLLMs and maps them to Parrott's hierarchical emotion model.
![Open-Vocabulary Emotion Tagging](images/INSETS-1.jpeg)
<div align="center">
<em>Figure 2: The open-vocabulary emotion tagging stage of INSETS.</em>
</div>

<br><br>
**Emotion Statement Construction**: Generates diverse statements covering four evaluation dimensions: sentiment polarity, emotion interpretation, scene context, and perception subjectivity.
![Emotion Statement Construction](images/INSETS-2.jpeg)
<div align="center">
<em>Figure 3: The emotion statement construction stage of INSETS.</em>
</div>

---

## INSETS-462k & MVEI Benchmark

- **INSETS-462k** is a large-scale corpus containing 462,369 emotion-centric statements derived from 17,716 images, annotated automatically via the INSETS pipeline.
- **MVEI** is a carefully human-refined benchmark comprising 3,086 high-quality image-statement pairs, designed for comprehensive evaluation of MLLMs' visual emotion intelligence.

<table>
<tr>
<td align="center">
<em>Table 1: Overall statistics.</em>
<img src="images/Statis-2.jpeg" width="95%">
<br>
</td>
<td align="center">
<img src="images/Statis-1.jpeg" width="100%">
<br>
<em>Figure 4: Detailed statistics of MVEI.</em>
</td>
</tr>
</table>

---

## Evaluation of MLLMs on MVEI
We evaluate a wide range of MLLMs on the MVEI benchmark. Below are the accuracy results across four affective cognitive dimensions:

<div align="center">
<em>Table 2: Accuracy evaluation of popular MLLMs on MVEI.</em>
</div>

![Main Results](images/Evaluation-1.jpeg)

<div align="center">
<em>Table 3: Comparison with human performance on MVEI.</em>
</div>

![Human Results](images/Evaluation-2.jpeg)


---

## Visualization of MVEI

<table>
<tr>
<td align="center">
<img src="images/MVEI-1.jpeg" width="100%">
<br>
<em>Figure 5: Correct sentiment polarity statements.</em>
</td>
<td align="center">
<img src="images/MVEI-2.jpeg" width="100%">
<br>
<em>Figure 6: Incorrect sentiment polarity statements.</em>
</td>
</tr>
</table>

<table>
<tr>
<td align="center">
<img src="images/MVEI-3.jpeg" width="100%">
<br>
<em>Figure 7: Correct emotion interpretation statements.</em>
</td>
<td align="center">
<img src="images/MVEI-4.jpeg" width="100%">
<br>
<em>Figure 8: Incorrect emotion interpretation statements.</em>
</td>
</tr>
</table>

<table>
<tr>
<td align="center">
<img src="images/MVEI-5.jpeg" width="100%">
<br>
<em>Figure 9: Correct scene context statements.</em>
</td>
<td align="center">
<img src="images/MVEI-6.jpeg" width="100%">
<br>
<em>Figure 10: Incorrect scene context statements.</em>
</td>
</tr>
</table>

<table>
<tr>
<td align="center">
<img src="images/MVEI-7.jpeg" width="100%">
<br>
<em>Figure 11: Correct perception subjectivity statements.</em>
</td>
<td align="center">
<img src="images/MVEI-8.jpeg" width="100%">
<br>
<em>Figure 12: Incorrect perception subjectivity statements.</em>
</td>
</tr>
</table>
