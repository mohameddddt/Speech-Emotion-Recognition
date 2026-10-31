---
title: Speech Emotion Recognition
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: Hear which of six emotions is in a spoken sentence
---

# Speech Emotion Recognition

Record or upload a short English sentence and see whether it sounds angry, disgusted, afraid, happy, neutral or sad.

A frozen [WavLM](https://huggingface.co/microsoft/wavlm-base-plus) speech encoder feeds a calibrated logistic-regression probe. It was trained on [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) and evaluated on actors it never heard.

Code, training and evaluation: [github.com/mohameddddt/Speech-Emotion-Recognition](https://github.com/mohameddddt/Speech-Emotion-Recognition)

CREMA-D is made available under the Open Database License. Its example clips are included here with attribution: Cao et al., *CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset*, IEEE Transactions on Affective Computing, 2014.
