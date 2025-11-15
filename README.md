# Noise Simulation for Image Denoising

This repository provides a compact pipeline to generate **noise–clean image pairs** for training image denoising models. Clean images from the **FFHQ dataset** are processed with realistic degradations—including blur, contrast reduction, radial shading, grain noise, and texture distortion—to create challenging noisy counterparts.

The objective is to provide high-quality paired data for downstream tasks such as **SCUNet/Restormer training**, **UDC image restoration**, and general denoising research.

## Usage


Run: python3 oil.py

Noisy images will be saved to:
Maindir/train_noisy/

## Features
- Grayscale + blur  
- Darkening + contrast reduction  
- Radial illumination falloff  
- Grainy noise  
- Oil-painting texture artifacts  

These degradations help simulate realistic noise patterns for supervised denoising tasks.
