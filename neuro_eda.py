import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import streamlit as st

@st.cache_data
def load_nifti_images_from_folder(folder_path):
    images = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.nii'):
            file_path = os.path.join(folder_path, filename)
            img_data = nib.load(file_path).get_fdata()
            images.append((filename, img_data))
    return images

def plot_sample_slices(images, masks, num_samples=3):
    for i in range(min(num_samples, len(images))):
        img_name, img_data = images[i]
        mask_name, mask_data = masks[i]

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(img_data[:, :, img_data.shape[2] // 2], cmap='gray')
        plt.title(f'Image: {img_name}')
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.imshow(mask_data[:, :, mask_data.shape[2] // 2], cmap='hot', alpha=0.5)
        plt.title(f'Mask: {mask_name}')
        plt.colorbar()

        plt.show()


# Compute statistics
def compute_statistics_for_images(images):
    stats = []
    for name, data in images:
        mean_intensity = np.mean(data)
        std_intensity = np.std(data)
        stats.append((name, mean_intensity, std_intensity))
    return stats

# Plot image statistics
def plot_image_statistics(image_stats):
    names = [stat[0] for stat in image_stats]
    means = [stat[1] for stat in image_stats]
    stds = [stat[2] for stat in image_stats]

    x = range(len(names))

    plt.figure(figsize=(14, 7))
    plt.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color='skyblue', ecolor='black')
    plt.xticks(x, names, rotation=90)
    plt.xlabel('Image')
    plt.ylabel('Mean Intensity')
    plt.title('Mean Intensity of Images with Standard Deviation')
    plt.tight_layout()
    plt.show()


# Compute statistics
def compute_statistics_for_masks(masks):
    stats = []
    for name, data in masks:
        coverage = np.sum(data > 0) / data.size
        stats.append((name, coverage * 100))
    return stats

# Plot mask coverage
def plot_mask_coverage(mask_stats):
    names = [stat[0] for stat in mask_stats]
    coverages = [stat[1] for stat in mask_stats]

    x = range(len(names))

    plt.figure(figsize=(14, 7))
    plt.bar(x, coverages, color='lightcoral')
    plt.xticks(x, names, rotation=90)
    plt.xlabel('Mask')
    plt.ylabel('Coverage (%)')
    plt.title('Mask Coverage')
    plt.tight_layout()
    plt.show()


def plot_histograms(images, masks):
    all_img_data = np.concatenate([data.ravel() for _, data in images])
    all_mask_data = np.concatenate([data.ravel() for _, data in masks])

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(all_img_data, bins=100, color='gray')
    plt.title('Histogram of Image Intensities')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(all_mask_data, bins=2, color='red')
    plt.title('Histogram of Mask Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    plt.show()


def plot_overlay_for_samples(images, masks, num_samples=3):
    for i in range(min(num_samples, len(images))):
        img_name, img_data = images[i]
        mask_name, mask_data = masks[i]

        plt.figure(figsize=(8, 8))
        plt.imshow(img_data[:, :, img_data.shape[2] // 2], cmap='gray')
        plt.imshow(mask_data[:, :, mask_data.shape[2] // 2], cmap='hot', alpha=0.5)
        plt.title(f'{img_name} with {mask_name} Overlay')
        plt.colorbar()
        plt.show()




