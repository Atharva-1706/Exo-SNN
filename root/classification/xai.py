import numpy as np
import torch


def compute_1d_gradcam(model, local_tensor, global_tensor, tab_tensor, target_class=1):
    """Compute Grad-CAM from the final local convolution activations."""
    model.eval()
    activations = []
    gradients = []

    target_layer = model.local_conv[1].net[0]

    def forward_hook(_module, _inputs, output):
        activations.append(output)

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0])

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(global_tensor, local_tensor, tab_tensor)
        score = logits[0, target_class]
        score.backward()
        if not activations or not gradients:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients")

        act = activations[0][0]       # [C, L]
        grad = gradients[0][0]        # [C, L]
        weights = grad.mean(dim=1, keepdim=True)
        cam = torch.relu((weights * act).sum(dim=0))
        cam = torch.nn.functional.interpolate(
            cam[None, None, :], size=local_tensor.shape[-1], mode="linear", align_corners=False
        )[0, 0]
        saliency = cam.detach().cpu().numpy()
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        return saliency
    finally:
        h1.remove()
        h2.remove()
