from __future__ import annotations

import numpy as np
import numpy.typing as npt


def reward_MSE(
    ground_truth_rewards: npt.NDArray[np.float64],
    predicted_rewards: npt.NDArray[np.float64],
) -> float:
    """Compute the mean squared error between the ground truth rewards and the
    predicted rewards.

    Args:
        ground_truth_rewards (np.ndarray): Nx1 array of ground truth rewards.
        predicted_rewards (np.ndarray): Nx1 array of predicted rewards.

    Returns:
        float: The mean squared error between the ground truth rewards and the predicted rewards.
    """
    # Ensure inputs are numpy arrays
    ground_truth_rewards = np.asarray(ground_truth_rewards)
    predicted_rewards = np.asarray(predicted_rewards)

    # Check if the shapes of the inputs match
    if ground_truth_rewards.shape != predicted_rewards.shape:
        raise ValueError(
            "The shape of ground truth rewards and predicted rewards must match."
        )

    # Calculate and return the mean squared error
    return np.mean((ground_truth_rewards - predicted_rewards) ** 2)
