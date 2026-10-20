# Verified sources and attribution

Checked 2026-10-20. Primary sources were read for metadata, licensing, API behavior and preprocessing.

1. MedMNIST official site: https://medmnist.com/ — educational benchmark, standardized splits, license, clinical-use disclaimer.
2. Official metadata: https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py — PathMNIST label order; train 89996, val 10004, test 7180; version 3.0.2; CC BY 4.0; archive URL/checksum.
3. Official API README: https://github.com/MedMNIST/MedMNIST — `PathMNIST`, `split`, `size`, `download=True`; six arrays; Zenodo official distribution.
4. Official PathMNIST archive: https://zenodo.org/records/10519652/files/pathmnist.npz?download=1 — accessed automatically through the package. Expected MD5 from metadata: `a8b06965200029087d5bd730944a56c1`.
5. Original data: Kather, Halama, Marx, “100,000 histological images of human colorectal cancer and healthy tissue”: https://zenodo.org/records/1214456 — original patch construction, color normalization, tissue categories and separate 7K cohort. Source archive describes 50 patients in the 7K cohort with no overlap with the 100K cohort. This is a source-reported property, not this project's independent patient-ID audit.
6. MedMNIST v2 paper: Yang et al., Scientific Data (2023), https://doi.org/10.1038/s41597-022-01721-8; author manuscript https://arxiv.org/abs/2110.14795 . MedMNIST's requested citation. The publisher page was not accessible in this session; project metadata and official repository were used for implementation facts.
7. Kather et al., “Predicting survival from colorectal cancer histology slides using deep learning: A retrospective multicenter study,” PLOS Medicine (2019), https://doi.org/10.1371/journal.pmed.1002730 — original study attribution. This portfolio task does not reproduce survival prediction.
8. torchvision ResNet18: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html — `IMAGENET1K_V1`, bilinear resize to 256, center crop 224, mean [0.485,0.456,0.406], std [0.229,0.224,0.225]. The code calls the preset instead of reimplementing it.
9. National Cancer Institute H&E definition: https://www.cancer.gov/publications/dictionaries/cancer-terms/def/hematoxylin-and-eosin-staining — plain-language stain explanation.
10. CC BY 4.0: https://creativecommons.org/licenses/by/4.0/ . Credit the creators, link the license, and identify changes. MedMNIST upstream code: https://github.com/MedMNIST/MedMNIST/blob/main/LICENSE .

Suggested data attribution on exported figures: “PathMNIST / MedMNIST (Yang et al.), derived from NCT-CRC-HE-100K and CRC-VAL-HE-7K (Kather et al.), CC BY 4.0. Images downsampled by MedMNIST; visualization/preprocessing as documented in this project.”

No dataset or pretrained weight redistribution is included. The downloaded archive's hash and package metadata are saved by each run. License facts are reproduced from the upstream documentation, not a claim of legal review.
