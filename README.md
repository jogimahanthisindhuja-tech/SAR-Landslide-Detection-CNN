# SAR-Landslide-Detection-CNN
This project implements landslide detection using InSAR deformation data and a CNN model. It integrates radar radiation patterns to enhance data reliability, performs patch-based classification, and generates prediction maps highlighting potential landslide-prone regions for analysis and visualization.
# Radiation Pattern-Aware InSAR Framework for Landslide Detection

##  Overview
Landslides are one of the most destructive natural hazards, causing severe damage to infrastructure, the environment, and human life—especially in mountainous and geologically unstable regions. This project presents an advanced framework that integrates antenna design, SAR signal processing, interferometric analysis, and machine learning to improve landslide detection accuracy.

##  Project Objective
The goal of this project is to develop a **radiation pattern-aware InSAR-based system** that enhances the accuracy of ground deformation monitoring and enables reliable detection of landslide-prone areas.

##  Key Concepts

- **Synthetic Aperture Radar (SAR):** Enables high-resolution imaging under all weather conditions, day and night.
- **Interferometric SAR (InSAR):** Detects minute ground displacements with millimeter-level accuracy using phase differences.
- **Radiation Pattern Integration:** Improves SAR signal modeling by incorporating realistic antenna characteristics.
- **Machine Learning (CNN):** Automates detection and classification of landslide-prone regions.

##  Methodology

### 1. Antenna Design
- A **Rectangular Microstrip Patch Antenna** operating at **3.5 GHz** is designed.
- Simulation is performed using **HFSS**.
- Key parameters extracted:
  - Gain
  - Beamwidth
  - Side-lobe levels

### 2. SAR Signal Modeling
- Traditional SAR assumes ideal antenna behavior.
- This project incorporates **realistic radiation patterns** as a spatial weighting function.
- Enhances accuracy in SAR image formation.

### 3. InSAR Processing
- Multiple SAR images are processed to:
  - Compute phase differences
  - Estimate ground deformation
  - Identify slope instability

### 4. Machine Learning Integration
- **Convolutional Neural Networks (CNNs)** are used to:
  - Analyze SAR/InSAR data
  - Detect deformation patterns
  - Classify landslide-prone regions

##  Workflow

1. Antenna Simulation (HFSS)
2. Radiation Pattern Extraction
3. SAR Image Formation with Weighting
4. InSAR Deformation Analysis
5. CNN-Based Landslide Detection

##  Expected Outcomes

- Improved SAR image realism
- Enhanced deformation detection accuracy
- Automated landslide risk classification
- Reliable early warning support system

##  Tools & Technologies

- HFSS (High Frequency Structure Simulator)
- SAR & InSAR Processing Techniques
- Python (for data processing & ML)
- TensorFlow / PyTorch (for CNN models)

## 📂 Project Structure (Example)
