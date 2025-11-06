# Action-Aware LLMs for Realistic Simulation of Local Government Meetings

This repository accompanies the paper:  
**_“Action-Aware LLMs for Realistic Simulation of Local Government Meetings”_**, which introduces a multimodal framework for transforming real-world meeting videos into **speaker-linked, metadata-enriched transcripts** and **fine-tuned language models** capable of simulating realistic deliberative interactions.

---

## Overview

Local government meetings, such as school boards, city councils, and appellate courts, are deliberative arenas where participants debate, negotiate, and make consequential decisions.  
This project presents an **action-aware** modeling pipeline that captures both *who* is speaking and *what actions* they are performing—enabling more realistic, role-sensitive simulations of institutional discourse.

The system combines:

- **Speaker diarization** to detect and label individual speakers from meeting recordings  
- **Automatic speech recognition (ASR)** for high-quality transcripts  
- **Action and metadata extraction** for structured context  
- **Fine-tuned LLMs** trained to reproduce realistic dialogue dynamics and role consistency

---

## Framework Overview

The following figure provides an overview of the end-to-end workflow, from meeting recordings to simulated deliberation outputs.

<p align="center">
  <img src="Figures/main_figure.png" alt="Overview of the Action-Aware LLM framework" width="800"/>
</p>

**Figure 1.** Overview of the Action-Aware LLM framework for realistic simulation of local government meetings.

---

## Diarization Process

The diarization pipeline links anonymized ASR outputs to speaker identities based on timestamp alignment and meeting metadata. This process is critical for creating consistent, speaker-linked transcripts.

<p align="center">
  <img src="Figures/diarization.png" alt="Speaker diarization and transcript alignment process" width="700"/>
</p>

**Figure 2.** Workflow for generating speaker-linked transcripts from meeting audio and metadata.

---

## Module Descriptions

### `Analysis/`
Contains evaluation notebooks for assessing model performance across key metrics:

- **`analysis_fool_rates.ipynb`** – Computes and visualizes *Classifier Fool Rate (CFR)*, quantifying how often a model-generated utterance is indistinguishable from a real one.  
- **`analysis_perplexity.ipynb`** – Measures linguistic fluency using standard *perplexity* metrics.  
- **`analysis_speaker_attribution.ipynb`** – Evaluates identity consistency via *Speaker Attribution Accuracy (SAA)*.

---

### `Data Processing/`
Implements the end-to-end multimodal diarization and transcription pipeline:

- **`zoomDiarization.py`** – Produces a time-stamped list of speaker turns along with identified speaker names extracted from Zoom meeting metadata.  
- **`whisperTranscription.py`** – Generates automatic speech recognition (ASR) transcripts using OpenAI’s Whisper model.  
- **`make_transcript.ipynb`** – Aligns anonymous speaker labels from `whisperTranscription.py` with identity-linked names from `zoomDiarization.py` by matching timestamps of utterances and speaker changes.

---

### `Datasets/`
Contains curated, speaker-linked deliberation datasets:

- **`Albermale/`**, **`DCAppeals/`**, and **`Waipa/`** – Transcripts from three real-world government forums: a school board, an appellate court, and a municipal council.  
- Each folder includes `.npy` transcript arrays with time-aligned, speaker-linked dialogue segments.  
- Files are named according to their corresponding YouTube videos. You can view each source video by visiting:  
  `https://www.youtube.com/watch?v={VIDEO_ID}`  
- **`dataset_example.ipynb`** – Demonstrates dataset structure, loading, and preprocessing steps.

---

### `Scripts/`
Core training and simulation scripts:

- **`train_agent.py`** – Fine-tunes large language models on speaker-linked transcripts to generate persona-consistent dialogue.  
- **`train_fool_rate.py`** – Trains models for *Classifier Fool Rate (CFR)* evaluation.  
- **`train_SAA.py`** – Trains models for *Speaker Attribution Accuracy (SAA)* evaluation.  
- **`simulate.py`** – Runs full deliberation simulations given meeting contexts and defined personas.  
- **`eval_test_set.py`**, **`eval_perplexity.py`** – Evaluate model outputs on held-out transcripts for accuracy and fluency.  
- **`utils.py`** – Shared utilities for data loading, preprocessing, and model management.

---

## Citation

If you use this code or dataset, please cite:

@inproceedings{TODO,
title={Action-Aware LLMs for Realistic Simulation of Local Government Meetings},
author={Scott Merrill and Shashank Srivastava},
booktitle={TODO},
year={2026}
}


---

## Contact

For questions or collaboration inquiries, please contact:  
**Scott Merrill** – [smerrill@unc.edu]  
or open an issue on GitHub.