from . import pcd_utils
from .datasets import ex, pt
import torch
import time
import numpy as np


def predict_pcd(pcd, model, device, threshold=0.5):
    selected_index = pcd_utils.select_points(pcd)
    # selected_index = np.arange(len(pcd.points))
    pcd_selected = pcd.select_by_index(selected_index)
    x, x_sampled, group_index, knn_index = pcd_utils.generate_model_data(pcd_selected, sample_size=512)

    x = torch.from_numpy(x).float()
    x_sampled = torch.from_numpy(x_sampled).float()
    group_index = torch.from_numpy(group_index).long()
    knn_index = torch.from_numpy(knn_index).long()

    x_batch, x_sampled_batch, group_index_batch, knn_index_batch, labeled_batch, mask_batch,\
        mask_sampled_batch = ex.collate_fn([(x, x_sampled, group_index, knn_index, [])])
    
    with torch.no_grad():
        x_batch = x_batch.to(device)
        x_sampled_batch = x_sampled_batch.to(device)
        group_index_batch = group_index_batch.to(device)
        knn_index_batch = knn_index_batch.to(device)
        mask_sampled_batch = mask_sampled_batch.to(device)
        x_logits = model(x_batch, x_sampled_batch, group_index_batch, knn_index_batch, mask_sampled_batch)
        pass
    x_logits = x_logits.squeeze(0)
    proba = torch.sigmoid(x_logits)
    pred_index = proba > threshold
    pred_index = pred_index.cpu().numpy()

    pred_index = selected_index[pred_index]

    return pred_index, proba.cpu().numpy()


def predict_pcd_pt(pcd, model, device, threshold=0.1, voxel_size=0):
    selected_index = pcd_utils.select_points(pcd)
    # selected_index = np.arange(len(pcd.points))
    pcd_selected = pcd.select_by_index(selected_index)

    if voxel_size > 0:
        pcd_selected, _, trace = pcd_selected.voxel_down_sample_and_trace(
            voxel_size=voxel_size, min_bound=pcd_selected.get_min_bound(), max_bound=pcd_selected.get_max_bound()
        )
        pass

    x, feat = pcd_utils.generate_model_data2(pcd_selected, sample_size=512)
    print("Generated model data size: ", x.shape, feat.shape)

    x = torch.from_numpy(x).float()
    feat = torch.from_numpy(feat).float()

    x_batch, feat_batch, _, offset_batch = pt.collate_fn([(x, feat, [])])
    
    with torch.no_grad():
        x_batch = x_batch.to(device)
        feat_batch = feat_batch.to(device)
        offset_batch = offset_batch.to(device)
        start = time.time()
        x_logits = model(x_batch, feat_batch, offset_batch)
        print("Time taken for forward: ", time.time() - start)
        pass
    x_logits = x_logits.squeeze(0)
    proba = torch.sigmoid(x_logits)
    pred_index = proba > threshold
    pred_index = pred_index.cpu().numpy()
    proba = proba.cpu().numpy()

    if voxel_size > 0:
        pred_index = np.where(pred_index)[0]
        pred_index = np.unique(np.concatenate([trace[i] for i in pred_index]))
        pred_index = selected_index[pred_index]
        pass
    else:
        pred_index = selected_index[pred_index]
        pass

    return pred_index, proba

def predict_pcd_batch(pcds, model, device, threshold=0.5, batch_size=32):
    pred_indices = []
    probas = []

    for i in range(0, len(pcds), batch_size):
        pcd_batch = pcds[i:i + batch_size]
        selected_indices = []
        batch_data = []
        for pcd in pcd_batch:
            
            selected_index = pcd_utils.select_points(pcd)
            selected_indices.append(selected_index)
            pcd_selected = pcd.select_by_index(selected_index)
            start = time.time()
            x, x_sampled, group_index, knn_index = pcd_utils.generate_model_data(pcd_selected, sample_size=512)
            x = torch.from_numpy(x).float()
            x_sampled = torch.from_numpy(x_sampled).float()
            group_index = torch.from_numpy(group_index).long()
            knn_index = torch.from_numpy(knn_index).long()
            batch_data.append((x, x_sampled, group_index, knn_index, []))
            
            pass
        
        x_batch, x_sampled_batch, group_index_batch, knn_index_batch, labeled_batch, mask_batch,\
            mask_sampled_batch = ex.collate_fn(batch_data)
        
        start = time.time()
        with torch.no_grad():
            x_batch = x_batch.to(device)
            x_sampled_batch = x_sampled_batch.to(device)
            group_index_batch = group_index_batch.to(device)
            knn_index_batch = knn_index_batch.to(device)
            mask_batch = mask_batch.to(device)
            mask_sampled_batch = mask_sampled_batch.to(device)
            x_logits = model(x_batch, x_sampled_batch, group_index_batch, knn_index_batch, mask_sampled_batch)
            pass
        proba = torch.sigmoid(x_logits)
        proba = proba * mask_batch
        pred_mask_batch = proba > threshold
        pred_mask_batch = pred_mask_batch.cpu().numpy()
        
        print("Time taken for predict batch: ", time.time() - start)
        for j, pred_mask in enumerate(pred_mask_batch):
            selected_index = selected_indices[j]
            pred_index = selected_index[pred_mask]
            pred_indices.append(pred_index)
            pass
        probas.append(proba.cpu().numpy())
        pass
    return pred_indices, np.concatenate(probas, axis=0)


if __name__ == "__main__":
    pass