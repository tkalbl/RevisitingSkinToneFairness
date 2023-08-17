# RevisitingSkinToneFairness
This repository contains the code to reproduce the experiments in the paper "Revisiting Skin Tone Fairness in Dermatological Lesion Classification", submitted to the [FAIMI workshop 2023](https://faimi-workshop.github.io/2023-miccai/) @ MICCAI 2023.

For full reproducability of the experiments with Deep Learning-based Healthy Skin Segmentation (DLHSS), the original skin tone (ITa) labels from [Kinyanjui et al.](https://link.springer.com/chapter/10.1007/978-3-030-59725-2_31) are necessary in the metadata.csv (data/.../metadata.csv).
Since they were not previously publsihed by the original authors, they are anonymized here.

The ITA labels from the Random Patch (RP) approach from [Bevan et Atapour-Abarghouei]()

## Results:
The results and their visualisations can be reproduced with the predictions folder and with the jupyter notebooks.

## Experimental Setup:
The experiments including the model training and grid search optimization of the MobileNetV2 can be seen in the python files.
