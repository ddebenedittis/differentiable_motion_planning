"""Adaptive gradient balancing for two-loss training."""


class AdaptiveGradientBalancer:
    """Strategy 5A: balance L_ocp and L_reg gradient magnitudes via EMA.

    Two probe backward passes per step to estimate gradient norms, then
    returns lambda_hat = lambda_0 * EMA(||grad_ocp||) / EMA(||grad_reg||).
    Caller must do the final combined backward after calling step().
    """

    def __init__(self, lambda_0=0.3, ema_decay=0.99, eps=1e-8):
        self.lambda_0 = lambda_0
        self.ema_decay = ema_decay
        self.eps = eps
        self.ema_ocp = None
        self.ema_reg = None

    def step(self, theta, loss_ocp, loss_reg):
        """Probe gradients and return adaptive lambda_hat.

        Args:
            theta: the parameter tensor (theta for softmax simplex)
            loss_ocp: scalar tensor for the OCP loss
            loss_reg: scalar tensor for the regularizer loss

        Returns:
            lambda_hat: scalar weight for the regularizer
        """
        # Probe L_ocp gradient
        loss_ocp.backward(retain_graph=True)
        g_ocp = theta.grad.abs().max().item()
        theta.grad = None

        # Probe L_reg gradient
        loss_reg.backward(retain_graph=True)
        g_reg = theta.grad.abs().max().item()
        theta.grad = None

        # Update EMA
        if self.ema_ocp is None:
            self.ema_ocp, self.ema_reg = g_ocp, g_reg
        else:
            d = self.ema_decay
            self.ema_ocp = d * self.ema_ocp + (1 - d) * g_ocp
            self.ema_reg = d * self.ema_reg + (1 - d) * g_reg

        return self.lambda_0 * self.ema_ocp / (self.ema_reg + self.eps)
