import numpy as np
import cv2
import matplotlib.pyplot as plt

from typing import List, Dict

# Constants
INPUT_PATH = "./input_images/"
OUTPUT_PATH = "./output/"

STANDARD_IMG_PATH = INPUT_PATH + "boat_col.png"
INOISE_IMG_PATH = INPUT_PATH + "boat_col_inoise.png"
NOISE_IMG_PATH = INPUT_PATH + "boat_col_noise.png"

MASKS = [3, 5, 7]
WEIGHTS = [1, 2, 5, 8, 10]


def save_image(image: np.array, image_name: str, file_name: str) -> None:
    """
    Saves image to the output directory with the specified name.

    Args:
        image (np.array): Image to be saved.
        image_name (str): Title of the image.
        file_name (str): Name of the file to save the image as.
    """
    plt.figure(figsize=(12, 8))
    plt.title(image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    plt.imshow(image, vmin=0, vmax=255)
    plt.xticks([]), plt.yticks([])
    plt.savefig(OUTPUT_PATH + file_name, bbox_inches="tight", pad_inches=0.1)
    plt.close()


def save_image_histogram(image: np.array, image_name: str, file_name: str) -> None:
    """
    Saves histogram of the image to the output directory with the specified name.

    Args:
        image (np.array): Image to be saved.
        image_name (str): Title of the image.
        file_name (str): Name of the file to save the image as.
    """
    image_histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    image_histogram.flatten()
    plt.figure(figsize=(12, 8))
    plt.title(image_name)
    plt.plot(image_histogram, color="black")
    plt.xlim([0, 256])
    plt.savefig(OUTPUT_PATH + file_name, bbox_inches="tight", pad_inches=0.1)
    plt.close()


def calculate_psnr(first_image: np.array, second_image: np.array) -> float:
    """
    Calculates PSNR between two images.

    Args:
        first_image (np.array): First image.
        second_image (np.array): Second image.
    Returns:
        float: PSNR value.
    """
    imax = 255.0**2
    mse = (
        (first_image.astype(np.float64) - second_image) ** 2
    ).sum() / first_image.size
    return 10.0 * np.log10(imax / mse)


def gaussian_filter(image: np.array, masks: List[int]) -> Dict[int, np.array]:
    """
    Applies Gaussian filter to the image with specified masks.
    Args:
        image (np.array): Image to be filtered.
        masks (List[int]): List of mask sizes.
    Returns:
        Dict[int, np.array]: Dictionary with mask size as key and filtered image as value.
    """
    results = {}
    for mask in masks:
        result = cv2.GaussianBlur(image, (mask, mask), 0)
        results[mask] = result
    return results


def median_filter(image: np.array, masks: List[int]) -> Dict[int, np.array]:
    """
    Applies median filter to the image with specified masks.
    Args:
        image (np.array): Image to be filtered.
        masks (List[int]): List of mask sizes.
    Returns:
        Dict[int, np.array]: Dictionary with mask size as key and filtered image as value.
    """
    results = {}
    for mask in masks:
        result = cv2.medianBlur(image, mask)
        results[mask] = result
    return results


def laplace_filter(image: np.array, weight: int) -> np.array:
    """
    Applies Laplace filter to the image with specified weight.
    Args:
        image (np.array): Image to be filtered.
        weight (int): Weight of the filter.
    Returns:
        np.array: Filtered image.
    """
    gauss = cv2.GaussianBlur(image, (3, 3), 0)
    laplace = cv2.Laplacian(gauss, cv2.CV_64F)
    image = image.astype(np.float64)
    laplace = cv2.addWeighted(image, 1, laplace, -weight, 0)
    laplace = np.clip(laplace, 0, 255)
    return laplace.astype(np.uint8)


if __name__ == "__main__":
    # Task 1
    print("Zadanie 1")
    # Load images
    inoise_image = cv2.imread(INOISE_IMG_PATH, cv2.IMREAD_UNCHANGED)
    noise_image = cv2.imread(NOISE_IMG_PATH, cv2.IMREAD_UNCHANGED)

    # Filter images
    inoise_gaussian = gaussian_filter(inoise_image, MASKS)
    noise_gaussian = gaussian_filter(noise_image, MASKS)
    inoise_median = median_filter(inoise_image, MASKS)
    noise_median = median_filter(noise_image, MASKS)

    standard_image = cv2.imread(STANDARD_IMG_PATH, cv2.IMREAD_UNCHANGED)

    pnsr_values = {}

    save_image(
        inoise_image,
        "SZUM IMPULSOWY",
        "inoise_image.png",
    )

    save_image(
        noise_image,
        "SZUM GAUSSOWSKI",
        "noise_image.png",
    )

    # Calculate PSNR for each mask and save images
    for i in MASKS:
        # Gaussian filter
        save_image(
            inoise_gaussian[i],
            f"SZUM IMPULSOWY PO FILTRZE GAUSSA - MASKA {i}x{i}",
            f"inoise_gaussian_{i}.png",
        )
        print(f"ZAPISANO - inoise_gaussian_{i}.png")
        inoise_gaussian_pnsr = calculate_psnr(standard_image, inoise_gaussian[i])
        pnsr_values[f"inoise_gaussian_{i}"] = inoise_gaussian_pnsr
        print(
            f"PSNR - SZUM IMPULSOWY PO FILTRZE GAUSSA - MASKA {i}x{i}: ",
            inoise_gaussian_pnsr,
        )

        save_image(
            noise_gaussian[i],
            f"SZUM GAUSSOWSKI PO FILTRZE GAUSSA - MASKA {i}x{i}",
            f"noise_gaussian_{i}.png",
        )
        print(f"ZAPISANO - noise_gaussian_{i}.png")
        noise_gaussian_pnsr = calculate_psnr(standard_image, noise_gaussian[i])
        pnsr_values[f"noise_gaussian_{i}"] = noise_gaussian_pnsr
        print(
            f"PSNR - SZUM GAUSSOWSKI PO FILTRZE GAUSSA - MASKA {i}x{i}: ",
            noise_gaussian_pnsr,
        )

        # Median filter
        save_image(
            inoise_median[i],
            f"SZUM IMPULSOWY PO FILTRZE MEDIANOWYM - MASKA {i}x{i}",
            f"inoise_median_{i}.png",
        )
        print(f"ZAPISANO - inoise_median_{i}.png")
        inoise_median_pnsr = calculate_psnr(standard_image, inoise_median[i])
        pnsr_values[f"inoise_median_{i}"] = inoise_median_pnsr
        print(
            f"PSNR - SZUM IMPULSOWY PO FILTRZE MEDIANOWYM - MASKA {i}x{i}: ",
            inoise_median_pnsr,
        )

        save_image(
            noise_median[i],
            f"SZUM GAUSSOWSKI PO FILTRZE MEDIANOWYM - MASKA {i}x{i}",
            f"noise_median_{i}.png",
        )
        print(f"ZAPISANO - noise_median_{i}.png")
        noise_median_pnsr = calculate_psnr(standard_image, noise_median[i])
        pnsr_values[f"noise_median_{i}"] = noise_median_pnsr
        print(
            f"PSNR - SZUM GAUSSOWSKI PO FILTRZE MEDIANOWYM - MASKA {i}x{i}: ",
            noise_median_pnsr,
        )
    # Save PSNR values to a text file
    with open(OUTPUT_PATH + "pnsr_values.txt", "w") as f:
        for key, value in pnsr_values.items():
            f.write(f"{key}: {round(value, 2)}\n")
    print("ZAPISANO - pnsr_values.txt")

    # Task 2
    print("Zadanie 2")

    standard_image = cv2.imread(STANDARD_IMG_PATH, cv2.IMREAD_UNCHANGED)

    # Convert to YCrCb color space and apply histogram equalization
    standard_image_YCrCb = cv2.cvtColor(standard_image, cv2.COLOR_BGR2YCrCb)
    standard_image_YCrCb[:, :, 0] = cv2.equalizeHist(standard_image_YCrCb[:, :, 0])
    standard_image_YCrCb = cv2.cvtColor(standard_image_YCrCb, cv2.COLOR_YCrCb2BGR)
    save_image(
        standard_image,
        "STANDARDOWY OBRAZ",
        "standard_image.png",
    )
    print("ZAPISANO - standard_image.png")
    save_image(
        standard_image_YCrCb,
        "OBRAZ PO WYRÓWNANIU HISTOGRAMU",
        "standard_image_YCrCb.png",
    )
    print("ZAPISANO - standard_image_YCrCb.png")
    save_image_histogram(
        standard_image,
        "HISTOGRAM OBRAZU PRZED WYRÓWNANIEM",
        "standard_image_histogram.png",
    )
    print("ZAPISANO - standard_image_histogram.png")
    save_image_histogram(
        standard_image_YCrCb,
        "HISTOGRAM OBRAZU PO WYRÓWNANIU",
        "standard_image_YCrCb_histogram.png",
    )
    print("ZAPISANO - standard_image_YCrCb_histogram.png")

    # Task 3
    print("Zadanie 3")

    standard_image = cv2.imread(STANDARD_IMG_PATH, cv2.IMREAD_COLOR)

    # Apply Laplace filter with different weights
    for weight in WEIGHTS:
        laplace_image = laplace_filter(standard_image, weight)
        save_image(
            laplace_image.astype(np.uint8),
            f"OBRAZ PO FILTRACJI LAPLASA - WAGA {weight}",
            f"laplace_{weight}.png",
        )
        print(f"ZAPISANO - laplace_{weight}.png")
