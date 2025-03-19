import os
import torch
import loguru
from .. import datasets
from .. import models
import numpy as np


logger = loguru.logger

def run_train(data_path, device, num_epochs=100):
    torch.autograd.set_detect_anomaly(True)
    train_dataset = datasets.match.PointMatchDataset(os.path.join(data_path), size=64, voxel_size=10)
    val_dataset = datasets.match.PointMatchDataset(os.path.join(data_path), is_test=True, voxel_size=10)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, collate_fn=datasets.match.collate_fn, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, collate_fn=datasets.match.collate_fn, num_workers=0)

    model = models.match.PointMatcher(6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    warmup_epochs = 20

    model.to(device)

    for iepoch in range(num_epochs):
        model.train()
        if iepoch < warmup_epochs:
            optimizer.param_groups[0]['lr'] = 1e-5 * min((iepoch+1) / warmup_epochs, 1.0)
            pass

        for i, batch in enumerate(train_loader):
            batch = [x.to(device) for x in batch]
            x0, feat0, offset0, x1, feat1, offset1, T = batch
            pred_T = model(x0, feat0, offset0, x1, feat1, offset1)
            loss = model.compute_loss(pred_T, T)
            loss = loss.mean()
            print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pass

        model.eval()

        losses = []

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                batch = [x.to(device) for x in batch]
                x0, feat0, offset0, x1, feat1, offset1, T = batch
                pred_T = model(x0, feat0, offset0, x1, feat1, offset1)
                loss = model.compute_loss(pred_T, T)
                loss = loss.mean()
                print(f'val loss: {loss.item()}')
                losses.append(loss.item())
                pass

            logger.info(f'Epoch {iepoch}, val loss: {np.mean(losses)}, lr: {optimizer.param_groups[0]["lr"]}')
            pass
        
        if iepoch >= warmup_epochs:
            scheduler.step()
            pass
        pass

    output_path = os.path.join('.', 'output')
    os.makedirs(output_path, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(output_path, 'model.pth'))
    pass