def mse_loss_fn(pred: torch.Tensor, gt: torch.Tensor):
      """
      The MSE error
      Parameters:
      pred (torch.Tensor): The reconstructed output from the autoencoder.
      gt (torch.Tensor): The original input to the autoencoder.

      Returns:
      torch.Tensor: The mean squared error loss.
      """
      mse = (pred - gt)**2
      return mse.mean()
