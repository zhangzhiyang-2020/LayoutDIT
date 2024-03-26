# LayoutDIT
LayoutDIT is a layout-aware end-to-end document image translation (DIT) framework. It effectively incorporates the layout information into DIT in an end-to-end way and significantly improves the translation for document images of diverse domains and layouts/formats in our experiments.

Our paper [LayoutDIT: Layout-Aware End-to-End Document Image Translation with Multi-Step Conductive Decoder](https://aclanthology.org/2023.findings-emnlp.673/) has been accepted by EMNLP 2023.

DITrans is a new benchmark dataset for document image translation built with three document domains and fine-grained human-annotated labels, enabling the research on En-Zh DIT, reading order detection, and layout analysis. Note that the DITrans dataset can only be used for non-commercial research purposes. For scholars or organizations who want to use the dataset, please send an application via email to us (zhangzhiyang2020@ia.ac.cn). When submitting the application to us, please list or attach 1-2 of your publications in the recent 2 years to indicate that you (or your team) do research in the related research fields of document image processing or machine translation. At present, this dataset is only freely available to scholars in the above-mentioned fields. We will give you the download links and decompression passwords for the dataset after your letter has been received and approved.

# Dependency
```python
torch==1.8.1
transformers==4.30.0
jieba==0.42.1
nltk==3.8.1

# Training
