# utils
import os
from datetime import datetime
import random
import numpy as np
from scipy import stats

    
def calculate_error(real_images, estimated_images):
    error = torch.norm(real_images - estimated_images, p=2, dim=(1, 2, 3))  # Calculate the 2-norm error for each image
    return error.detach().cpu().numpy()

def save_images(real_images, save_dir,  **estimated_images):
    """
     Save the original image and multiple estimated images
    :param real_images: original image [batch_size, channels, height, width]
    :param save_dir: 
    :param estimated_images: Estimated image, in dictionary form, key is method name, value is estimated image tensor
    """
    # Get current date and create subfolders
    current_date = datetime.now().strftime("%Y%m%d%H")
    save_dir = os.path.join(save_dir, current_date)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save original image
    np.save(os.path.join(save_dir, f"real.npy"), real_images.cpu().numpy())

    # Save estimated image
    for method, images in estimated_images.items():
        np.save(os.path.join(save_dir, f"{method}.npy"), images.detach().cpu().numpy())
        

import matplotlib.pyplot as plt


def visualize_images(save_dir, idx, *methods):
    """
    Visualize the original sequence frames of the specified index and the results of estimation methods
    :param save_dir: 
    :param idx: 
    :param methods: Method list, optional values are ["real", "seqcsgm", "csgm", "mom", "e2e", "lasso"]
    """
    # Create a dictionary to store images at all time points
    images_dict = {method: np.load(os.path.join(save_dir, f"{method}.npy")) for method in methods}
    
    #  Obtain the number of time steps
    T = images_dict["seqcsgm"].shape[1]

    # Visualization
    plt.figure(figsize=(8, 3.7))
    for i, method in enumerate(methods):
        for t in range(T):
            subplot_index = i * T + t + 1
            plt.subplot(len(methods), T, subplot_index)
            image = images_dict[method][idx, t]
            image = normalize_images(image, target_range=(0, 1))
            if image.shape[0] == 1:  # If it is a single channel image
                plt.imshow(image[0], cmap='gray')
            else:  
                image = np.transpose(image, (1, 2, 0))  # [height, width, channels]
                plt.imshow(image)
            plt.axis("off")
            name_mapping = {"seqcsgm":"SeqCSGM","csgm":"CSGM", "mom":"MOM","e2e":"E2E", "lasso": "ModifyCS"}
            if t == 0:  # Display method name in the first column
                plt.text(
                    x=-0.1, 
                    y=0.5,
                    s=name_mapping.get(method, method.capitalize()),
                    rotation=0,
                    ha='right',
                    va='center',
                    fontsize=14,
                    transform=plt.gca().transAxes 
                )
    
    plt.subplots_adjust(left=0.1, wspace=0.01, hspace=0.01)
    plt.show()
    
def visualize_images2(save_dir, t, num, *methods):
    """
    Visualize the original sequence frames at a specified time and the results of estimation methods
    :param save_dir:
    :param t: 
    :param num: The number of images displayed should be less than the number of batches
    :param methods: Method list, optional values are ["real", "seqcsgm", "csgm", "mom", "e2e", "lasso"]
    """
    images_dict = {method: np.load(os.path.join(save_dir, f"{method}.npy")) for method in methods}
    
    bath_size = images_dict["seqcsgm"].shape[0]
    idxs = random.sample(range(bath_size), num)

    # Visualization
    plt.figure(figsize=(8, 3.7))
    for i, method in enumerate(methods):
        for j,idx in enumerate(idxs):
            subplot_index = i * num + j + 1
            plt.subplot(len(methods), num, subplot_index)
            image = images_dict[method][idx, t]
            image = normalize_images(image, target_range=(0, 1))
            if image.shape[0] == 1: 
                plt.imshow(image[0], cmap='gray')
            else:  
                image = np.transpose(image, (1, 2, 0))  #  [height, width, channels]
                plt.imshow(image)
            plt.axis("off")
            name_mapping = {"seqcsgm":"SeqCSGM","csgm":"CSGM", "mom":"MOM","e2e":"E2E", "lasso": "ModifyCS"}
            if j == 0:  # Display method name in the first column
                plt.text(
                    x=-0.1,  
                    y=0.5,
                    s=name_mapping.get(method, method.capitalize()),
                    rotation=0,
                    ha='right',
                    va='center',
                    fontsize=14,
                    transform=plt.gca().transAxes 
                )
    
    plt.subplots_adjust(left=0.1, wspace=0.01, hspace=0.01)
    plt.show()
    
    

def normalize_images(images, target_range=(-1, 1)):
    """
    Normalize image data to a specified range
    """
    min_val = np.min(images)
    max_val = np.max(images)
    normalized_images = (images - min_val) / (max_val - min_val)
    
    if target_range == (-1, 1):
        normalized_images = 2 * normalized_images - 1
    return normalized_images

    

def analyze_and_visualize_errors(save_dir, *methods):
    """
    Analyze and visualize the relationship between errors of different methods over time
    :param save_dir: 
    :param methods: Method list, optional values are ["real", "seqcsgm", "mom", "e2e", "lasso"]
    """
    seqcsgm_images = np.load(os.path.join(save_dir, "seqcsgm.npy"))
    batch_size, T, channels, height, width = seqcsgm_images.shape

    # Create a dictionary to store the errors of all methods
    error_dict = {method: np.zeros((batch_size, T)) for method in methods if method != "real"}

    # real images
    real_images = np.load(os.path.join(save_dir, "real.npy"))
    real_images = normalize_images(real_images, target_range=(0, 1))

    # Calculate the error of each method
    for method in methods:
        if method != "real":
            estimated_images = np.load(os.path.join(save_dir, f"{method}.npy"))
            estimated_images = normalize_images(estimated_images, target_range=(0, 1))
            print(f"  Normalized range: min={np.min(estimated_images):.4f}, max={np.max(estimated_images):.4f}")
            for t in range(T):
                diff = real_images[:, t] - estimated_images[:, t]
                diff_reshaped = diff.reshape(batch_size, -1)
                error_dict[method][:, t] = np.linalg.norm(diff_reshaped, axis=1)

    mean_errors = {method: np.mean(error_dict[method], axis=0) for method in error_dict}
    std_errors = {method: np.std(error_dict[method], axis=0) for method in error_dict}   
    conf_intervals = {method: stats.t.interval(0.95, batch_size - 1, loc=mean_errors[method], scale=std_errors[method] / np.sqrt(batch_size)) for method in error_dict}
    
    # Visualization
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(error_dict)))
    method_colors = {m: colors[i] for i, m in enumerate(error_dict.keys())}
    
    linestyles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5, 1, 5))] 
    markers = ['o', 's', '^', 'D', 'p']  

    name_mapping = {"seqcsgm":"SeqCSGM","csgm":"CSGM", "mom":"MOM","e2e":"E2E", "lasso": "ModifyCS"}
    label = name_mapping.get(method, method.upper())
    for i, method in enumerate(error_dict.keys()):
        label = name_mapping.get(method, method.upper())
        linestyle = linestyles[i % len(linestyles)]
        marker = markers[i % len(markers)]
        ax.plot(
            mean_errors[method], 
            color=method_colors[method],
            lw=2,
            linestyle=linestyle, 
            marker=marker,
            markersize=6,
            label=label
        )
        
        ax.fill_between(
            range(T),
            mean_errors[method] - std_errors[method],
            mean_errors[method] + std_errors[method],
            color=method_colors[method],
            alpha=0.15
        )
        
        ax.errorbar(
            x=range(T),
            y=mean_errors[method],
            yerr=[
                mean_errors[method] - conf_intervals[method][0],
                conf_intervals[method][1] - mean_errors[method]
            ],
            fmt='none',
            ecolor=method_colors[method],
            elinewidth=1.5,
            capsize=4,
            capthick=1.5
        )

    ax.set_xlabel("Time Step", fontsize=20)
    ax.set_ylabel("Reconstruction Error", fontsize=20)
    ax.set_xticks(range(T))
    ax.set_xticklabels([f"{t}" for t in range(T)])
    ax.set_xlim(-0.3, T-1+0.1)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tick_params(axis='both', which='major', labelsize=14) 
    
    

#     ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
    ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.80), borderaxespad=0.,framealpha=0.4, ncol=2, fontsize=16)


    plt.xlabel("Time Step")
    plt.ylabel("Reconstruction Error")
#     plt.title("Error Analysis Over Time")
    plt.grid(True)
    plt.tight_layout() 
#     plt.savefig("ucf.pdf", format="pdf")
    plt.show()