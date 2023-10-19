import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from constants import COLORS


def overlay_heatmap_on_image(image, attentions, image_idx):
    """
    Overlay a heatmap on an image and plot the result.
    
    Parameters:
    - image: Torch tensor representing the image ([3, H, W])
    - attentions: Torch tensor representing the heatmap ([B, 1, H, W])
    - image_idx: Index to select the heatmap from the batch
    
    Returns:
    - Plots the original image, heatmap, and overlay side-by-side.
    """
    heatmap = attentions[0][image_idx].unsqueeze(0).squeeze().numpy()
    # Normalize the heatmap
    heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    # Convert image tensor to numpy and create RGBA heatmap
    image_np = image.numpy().transpose(1, 2, 0)
    heatmap_colored = np.zeros((*image_np.shape[:2], 4))
    heatmap_colored[..., 0] = heatmap_normalized   # Red channel
    heatmap_colored[..., 3] = heatmap_normalized   # Alpha channel
    # Overlay
    overlay = np.clip(image_np + heatmap_colored[..., :3] * heatmap_colored[..., 3:4], 0, 1)
    # Plot
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor(COLORS['background'])
    for a in ax:
        a.set_facecolor(COLORS['background'])
        a.axis("off")
    ax[0].imshow(image_np) 
    ax[0].set_title("Original Image\n", fontsize=13, color=COLORS['light_bg']); ax[0].axis("off")
    ax[1].imshow(heatmap, cmap='hot')
    ax[1].set_title("Attention Heatmap\n", fontsize=13, color=COLORS['light_bg']); ax[1].axis("off")
    ax[2].imshow(overlay)
    ax[2].set_title("Overlay\n", fontsize=13, color=COLORS['light_bg']); ax[2].axis("off")
    plt.tight_layout()
    plt.show()

def visualize_attention(unaug_img, dino_instance, device, patch_size=8, 
                        output_dir='./outputs', head_idx=0, image_idx=0):
    """
    Visualize the attention of a given image using a DINO model instance.
    Parameters:
    - unaug_img: Torch tensor representing the unaugmented image ([B, 3, H, W])
    - dino_instance: Instance of the DINO model
    - device: Device to which tensors are moved for model inference
    - patch_size: Size of the patches used in the DINO model
    - output_dir: Directory where outputs can be saved
    - image_idx: Index of the image to be visualized from the batch
    Returns:
    - Calls overlay_heatmap_on_image and visualizes the original image, attention, and overlay.
    """
    # Process the image
    img, bs = unaug_img[image_idx], unaug_img.shape[0]
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)
    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size
    # Get attentions from the model
    attentions = dino_instance.get_last_selfattention(img.to(device))
    nh = attentions.shape[1]  # number of head
    print(f'Number of heads: {nh}')
    print(f'Batch Size: {bs}')
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Visualize using the previously defined function
    overlay_heatmap_on_image(unaug_img[image_idx], attentions, head_idx)
