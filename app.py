import io

from matplotlib import pyplot as plt

import neuro_eda as neuro
import streamlit as st
import pandas as pd
import math
from pathlib import Path

disableWidgetStateDuplicationWarning = False

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Neuroinformatics',
    page_icon=':brain:', # This is an emoji shortcode. Could be a URL too.
)
# -----------------------------------------------------------------------------
# Draw the actual page
st.title(':brain: Brain Images EDA')

multi = '''
Browse Neuroscience data from [Kaggle](https://www.kaggle.com/datasets/rony32/neuroscience) website.Data contain both of
    
- Images in NIfTI Format
- Masks to selectively isolate or extract specific regions or structures from the image dataset.
  '''
st.markdown(multi)


# -----------------------------------------------------------------------------
st.text("")
st.header('Visualize Sample Slices')

# Define paths
images_folder = 'data/images'
masks_folder = 'data/masks'

# Load images and masks
images = neuro.load_nifti_images_from_folder(images_folder)
masks = neuro.load_nifti_images_from_folder(masks_folder)

# Plot sample slices
fig = neuro.plot_sample_slices(images, masks, num_samples=1)
st.pyplot(fig)


# -----------------------------------------------------------------------------
st.text("")
st.subheader('Calculate statistics for all images and masks')

image_stats = neuro.compute_statistics_for_images(images)
statImgFig = neuro.plot_image_statistics(image_stats)
st.pyplot(statImgFig)
st.caption("This is a mean intensities with error bars representing the standard deviation.")


mask_stats = neuro.compute_statistics_for_masks(masks)
statMskFig = neuro.plot_mask_coverage(mask_stats)
st.pyplot(statMskFig)
st.caption("This is the mask coverage percentages")

# -----------------------------------------------------------------------------
st.text("")
st.header('Histograms of Intensities')
histFig = neuro.plot_histograms(images, masks)
st.pyplot(histFig)
st.caption("Plot histograms of voxel intensities for all images and masks.")

# -----------------------------------------------------------------------------
st.text("")
st.header('Visualize Image and Mask Overlays')
ovrFig = neuro.plot_overlay_for_samples(images, masks, num_samples=3)
st.pyplot(ovrFig)
st.caption("Overlay masks on images for a few samples.")
