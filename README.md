# Code Sample: Neurophysiological Data Visualization Pipeline

This repository serves as a code sample to demonstrate my approach to writing clean Python code.

A condensed version of my first mini-project at the Laboratory of Sensory Processing. The goal was to build a robust pipeline for visualizing static Receiver Operating Characteristic (ROC) results, which quantify and compare individual neurons' selectivity in mice from the start to the end of an experimental session.

## Repo folders

### 1. `/brain_flatmaps`
This folder generates 2D "flatmaps" of the mouse brain based on the Swanson atlas. These visualizations are powerful tools for spatially mapping the magnitude and location of neuronal selectivity. Essentially, they provide an intuitive, visual answer to the question: "Where in the brain are the most significant changes happening?"

### 2. `/focality_indices`
This folder calculates the Focality Index (FI), a statistical metric that quantifies the distribution of neural activity. It provides a single, quantitative value to answer the question: "Is the significant neural activity concentrated in a few 'hotspot' regions, or is it distributed broadly across the entire brain?" 
