from . import pcd_utils
from . import datasets
import torch
import time


def predict_pcd(pcd, model, device):
    selected_index = pcd_utils.select_points(pcd)
    pcd_selected = pcd.select_by_index(selected_index)
    x, x_sampled, group_index, knn_index = pcd_utils.generate_model_data(pcd_selected, sample_size=256)

    x = torch.from_numpy(x).float()
    x_sampled = torch.from_numpy(x_sampled).float()
    group_index = torch.from_numpy(group_index).long()
    knn_index = torch.from_numpy(knn_index).long()

    x_batch, x_sampled_batch, group_index_batch, knn_index_batch, labeled_batch, mask_batch,\
        mask_sampled_batch = datasets.collate_fn([(x, x_sampled, group_index, knn_index, [])])
    
    with torch.no_grad():
        x_batch = x_batch.to(device)
        x_sampled_batch = x_sampled_batch.to(device)
        group_index_batch = group_index_batch.to(device)
        knn_index_batch = knn_index_batch.to(device)
        mask_sampled_batch = mask_sampled_batch.to(device)
        x_logits = model(x_batch, x_sampled_batch, group_index_batch, knn_index_batch, mask_sampled_batch)
        pass
    x_logits = x_logits.squeeze(0)

    pred_index = x_logits > 0
    pred_index = pred_index.cpu().numpy()

    pred_index = selected_index[pred_index]

    return pred_index


if __name__ == "__main__":
    pass