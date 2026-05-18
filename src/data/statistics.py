from math import sqrt

import torch


def compute_mean_std(loader: torch.utils.data.DataLoader) -> tuple[float, float]:
    pixel_sum = 0
    pixel_squared_sum = 0
    num_pixels = 0

    for batch in loader:
        images = batch[0]
        batch_size, channels, height, width = images.shape

        pixel_sum += images.sum(dim=(0, 2, 3))
        pixel_squared_sum += (images ** 2).sum(dim=(0, 2, 3))
        num_pixels += batch_size * height * width

    mean = pixel_sum / num_pixels
    std = torch.sqrt(pixel_squared_sum / num_pixels - mean ** 2)

    return mean, std

def find_best_thresholds(y_true, y_prob):
    """
    y_true: [N, C] binary labels
    y_prob: [N, C] sigmoid probabilities
    """
    thresholds = []

    for c in range(y_true.shape[1]):
        best_threshold = 0.5
        best_f1 = -1.0

        for threshold in np.linspace(0.01, 0.99, 99):
            y_pred_c = (y_prob[:, c] >= threshold).astype(int)
            f1 = f1_score(y_true[:, c], y_pred_c, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        thresholds.append(best_threshold)

    return np.array(thresholds)