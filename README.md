<h1 align="center">UniPCGC: Towards Practical Point Cloud Geometry Compression via an Efficient Unified Approach</h1>

<p align="center">
    <strong>Kangli Wang</strong><sup>1</sup>, <strong>Wei Gao</strong><sup>1,2*</sup><br>
    (<em>* Corresponding author</em>)
</p>

<p align="center">
    <sup>1</sup>SECE, Peking University<br>
    <sup>2</sup>Peng Cheng Laboratory, Shenzhen, China
</p>

## 📣 News
- [24-12-09] Our paper has been accepted to AAAI 2025.

## 📌 Introduction

We propose an efficient unified point cloud geometry compression framework UniPCGC. It is a lightweight framework that supports lossy compression, lossless compression, variable rate and variable complexity. First, we introduce the Uneven 8-Stage Lossless Coder (UELC) in the lossless mode, which allocates more computational complexity to groups with higher coding difficulty, and merges groups with lower coding difficulty. Second, Variable Rate and Complexity Module (VRCM) is achieved in the lossy mode through joint adoption of a rate modulation module and dynamic sparse convolution. Finally, through the dynamic combination of UELC and VRCM, we achieve lossy compression, lossless compression, variable rate and complexity within a unified framework. Compared to the previous state-of-the-art method, our method achieves a compression ratio (CR) gain of 8.1\% on lossless compression, and a Bjontegaard Delta Rate (BD-Rate) gain of 14.02\% on lossy compression, while also supporting variable rate and variable complexity.

<div align="center">
<img src="assets/unipcgc.jpg" width = 75% height = 75%/>
<br>
Ilustration of the proposed UniPCGC framework. 
</div>

## 🔎 Contact
If your have any comments or questions, feel free to contact [kangliwang@stu.pku.edu.cn](kangliwang@stu.pku.edu.cn).

## 👍 Acknowledgement
Thanks for their awesome works ([PCGCv2](https://github.com/NJUVISION/PCGCv2) and [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine)).

## 📘 Citation
Please consider citing our work as follows if it is helpful.
```bibtex
@inproceedings{wang2025unipcgc,
  title={UniPCGC: Towards Practical Point Cloud Geometry Compression via An Efficient Unified Approach},
  author={Wang, Kangli and Gao, Wei},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
```
