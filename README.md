Noise Simulation for Image Denoising

This repository provides a simple pipeline to generate noise–clean image pairs for training denoising models. We use the FFHQ dataset as clean input images and apply several realistic degradations—blur, contrast reduction, radial shading, grain noise, and texture distortion—to create noisy counterparts.

The goal is to prepare high-quality paired data for downstream tasks such as SCUNet/Restormer training, UDC image restoration, and general image denoising.

Usage

Place clean images in:

Maindir/train/


Run:

python3 oil.py


Noisy images will be saved to:

Maindir/train_noisy/

Features

Grayscale + blur

Darkening + contrast reduction

Radial illumination falloff

Grainy noise

Oil-painting texture artifacts

These simulated degradations provide challenging noisy inputs for supervised denoising.
