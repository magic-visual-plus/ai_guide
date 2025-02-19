import os
import torch
from . import datasets
from . import models


def run_train(data_path, device, num_epochs=100):
    train_dataset = datasets.PointNetDataset(os.path.join(data_path, 'train'), size=128)
    val_dataset = datasets.PointNetDataset(os.path.join(data_path, 'val'))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=datasets.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=datasets.collate_fn)

    model = models.PointNetEx(input_size=6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)
    warmup_epochs = 100

    model.to(device)

    for iepoch in range(num_epochs):
        model.train()
        if iepoch < warmup_epochs:
            optimizer.param_groups[0]['lr'] = 1e-4 * min((iepoch+1) / warmup_epochs, 1.0)
            pass

        for i, (x, x_sampled, group_index, knn_index, labeled, mask, mask_sampled) in enumerate(train_loader):
            x = x.to(device)
            x_sampled = x_sampled.to(device)
            group_index = group_index.to(device)
            knn_index = knn_index.to(device)
            labeled = labeled.to(device)
            mask = mask.to(device)
            mask_sampled = mask_sampled.to(device)

            logits = model(x, x_sampled, group_index, knn_index, mask_sampled)
            proba = torch.sigmoid(logits)
            mask_high_proba = ((labeled == 1) & (proba > 0.9)) | ((labeled == 0) & (proba < 0.1))
            mask[mask_high_proba] = 0
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labeled, reduction='none', pos_weight=torch.tensor([10.0]).to(device))
            if mask.sum() == 0:
                continue
            loss = (loss * mask).sum() / mask.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pass

        model.eval()

        total_loss = 0
        total_count = 0
        ious = []

        with torch.no_grad():
            for i, (x, x_sampled, group_index, knn_index, labeled, mask, mask_sampled) in enumerate(val_loader):
                x = x.to(device)
                x_sampled = x_sampled.to(device)
                group_index = group_index.to(device)
                knn_index = knn_index.to(device)
                labeled = labeled.to(device)
                mask = mask.to(device)
                mask_sampled = mask_sampled.to(device)

                logits = model(x, x_sampled, group_index, knn_index, mask_sampled)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labeled, reduction='none')
                loss = (loss * mask).sum() / mask.sum()

                proba = torch.sigmoid(logits)
                pred = (proba>0.5).float()
                pred = pred * mask
                intersection = (pred * labeled).sum()
                union = (pred + labeled).sum() - intersection
                iou = intersection / union
                ious.append(iou.item())
                total_count += labeled.shape[0]
                total_loss += loss.item()
                pass

            iou = sum(ious) / len(ious)
            total_loss /= len(val_loader)

            print(f'Epoch {iepoch}, val loss: {loss.item()}, val iou: {iou}, lr: {optimizer.param_groups[0]["lr"]}')
            pass
        
        if iepoch >= warmup_epochs:
            scheduler.step()
            pass
        pass

    output_path = os.path.join('.', 'output')
    os.makedirs(output_path, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(output_path, 'model.pth'))
    pass