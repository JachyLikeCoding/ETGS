import matplotlib.pyplot as plt
import numpy as np
import os


def visualize_images(image, gt_image, iteration, tilt_id, image_name):
    """
    可视化渲染的图像和 GT 图像
    """
    # 将 tensor 转换为 numpy 数组并移动到 CPU
    image_np = image.detach().cpu().numpy().transpose(1, 2, 0)  # 假设图像格式为 (C, H, W)
    gt_image_np = gt_image.detach().cpu().numpy().transpose(1, 2, 0)

    # 可视化图像
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image_np)
    ax[0].set_title(f'Rendered Image (Iteration: {iteration}, Tilt ID: {tilt_id})')
    ax[0].axis('off')

    ax[1].imshow(gt_image_np)
    ax[1].set_title(f'GT Image: {image_name}')
    ax[1].axis('off')

    plt.show()


def visualize_images_with_mask(image, gt_image, visible_mask, iteration, tilt_id, image_name, save_path):
    network_output_np = np.clip(image.detach().cpu().numpy().transpose(1, 2, 0), 0.001, 0.999)
    gt_image_np = np.clip(gt_image.detach().cpu().numpy().transpose(1, 2, 0), 0.001, 0.999)
    visible_mask_np = visible_mask.detach().cpu().numpy().transpose(1, 2, 0)
    # print('visible_mask_np:', visible_mask_np.shape)
    plt.figure(figsize=(12, 4))

    # Display network output
    plt.subplot(1, 3, 1)
    plt.title(f'Rendered Image (Iteration: {iteration}, Tilt ID: {tilt_id}')
    plt.imshow(network_output_np, cmap='gray')  # Adjust indexing as needed
    plt.axis('off')

    # Display ground truth
    plt.subplot(1, 3, 2)
    plt.title(f'GT Image: {image_name}')
    plt.imshow(gt_image_np, cmap='gray')  # Adjust indexing as needed
    plt.axis('off')

    # Display the effective area for loss calculation
    plt.subplot(1, 3, 3)
    plt.title("Visible Mask")
    plt.imshow(visible_mask_np, cmap='gray')  # Adjust indexing as needed
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path,'visualize_images_with_mask.png'))
    plt.close()
