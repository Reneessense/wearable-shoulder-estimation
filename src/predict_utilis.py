# predict_utils.py
import numpy as np
import torch

def predict_by_batch(model, X_tensor, batch_size=256):
    """
    使用小批量进行模型预测，避免显存溢出。

    Args:
        model (torch.nn.Module): 已训练好的 PyTorch 模型。
        X_tensor (torch.Tensor): 输入数据张量，形状为 [N, seq_len, input_size]。
        batch_size (int): 批处理大小。

    Returns:
        numpy.ndarray: 所有预测结果，拼接后的形状为 [N, output_dim]。
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch_x = X_tensor[i:i+batch_size]
            batch_pred = model(batch_x).cpu().numpy()
            preds.append(batch_pred)
    return np.vstack(preds)
