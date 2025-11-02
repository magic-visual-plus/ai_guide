import os
import torch
import loguru
from ..datasets import pt
from ..models import pt as models
from tqdm import tqdm


logger = loguru.logger

def run_train(data_path, device, num_epochs=100):
    train_dataset = pt.PointTransformerDataset(os.path.join(data_path, 'train'), size=32, voxel_size=8)
    val_dataset = pt.PointTransformerDataset(os.path.join(data_path, 'val'), is_test=True, voxel_size=8)
    # train_dataset = datasets.PointNetDatasetPickled(os.path.join(data_path, 'train'), size=256)
    # val_dataset = datasets.PointNetDatasetPickled(os.path.join(data_path, 'val'), is_test=True)
    pos_weight = 2.0
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=3, shuffle=True, collate_fn=pt.collate_fn, num_workers=16)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=3, shuffle=False, collate_fn=pt.collate_fn, num_workers=16)

    model = models.PointTransformerPointceptSmall(input_size=6)
    # model = models.PointTransformerPointcept(input_size=6)
    if os.path.exists(os.path.join('.', 'output', 'model.pth')):
        # model.load_state_dict(torch.load(os.path.join('.', 'output', 'model.pth')))
        # logger.info('Loaded model from checkpoint')
        pass

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    warmup_epochs = 10

    model.to(device)

    for iepoch in range(num_epochs):
        model.train()
        if iepoch < warmup_epochs:
            optimizer.param_groups[0]['lr'] = 1e-3 * min((iepoch+1) / warmup_epochs, 1.0)
            pass

        for i, (x, feat, label, offset) in enumerate(tqdm(train_loader)):
            x = x.to(device)
            feat = feat.to(device)
            offset = offset.to(device)
            label = label.to(device)
            logits = model(x, feat, offset)
            proba = torch.sigmoid(logits)
            mask_low_proba = ((label == 1) & (proba <= 0.9)) | ((label == 0) & (proba > 0.1))
            # mask[mask_high_proba] = 0
            # loss = torch.nn.functional.binary_cross_entropy_with_logits(
            #     logits, labeled, reduction='none', pos_weight=torch.tensor([10.0]).to(device))
            # if mask.sum() == 0:
            #     continue
            # loss = (loss * mask).sum() / mask.sum()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, label, pos_weight=torch.tensor([pos_weight]).to(device), reduction='none')
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pass

        model.eval()

        total_loss = 0
        total_count = 0
        ious = []

        with torch.no_grad():
            for i, (x, feat, label, offset) in enumerate(val_loader):
                x = x.to(device)
                feat = feat.to(device)
                offset = offset.to(device)
                label = label.to(device)
                logits = model(x, feat, offset)
                proba = torch.sigmoid(logits)
                mask_low_proba = ((label == 1) & (proba <= 0.9)) | ((label == 0) & (proba > 0.1))
                # mask[mask_high_proba] = 0
                # loss = torch.nn.functional.binary_cross_entropy_with_logits(
                #     logits, labeled, reduction='none', pos_weight=torch.tensor([10.0]).to(device))
                # if mask.sum() == 0:
                #     continue
                # loss = (loss * mask).sum() / mask.sum()
                # pos_weight = (label.shape[0] - label.sum()) / label.sum()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, label, pos_weight=torch.tensor([pos_weight]).to(device), reduction='none')
                loss = loss.mean()
                proba = torch.sigmoid(logits)
                pred = (proba>0.5).float()
                intersection = (pred * label).sum()
                union = (pred + label).sum() - intersection
                iou = intersection / union
                ious.append(iou.item())
                total_count += label.shape[0]
                total_loss += loss.item()
                pass

            iou = sum(ious) / len(ious)
            total_loss /= len(val_loader)

            logger.info(f'Epoch {iepoch}, val loss: {loss.item()}, val iou: {iou}, lr: {optimizer.param_groups[0]["lr"]}')
            pass
        
        if iepoch >= warmup_epochs:
            scheduler.step()
            pass
        pass

    output_path = os.path.join('.', 'output')
    os.makedirs(output_path, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(output_path, 'model.pth'))
    pass