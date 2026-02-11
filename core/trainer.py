import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    # tqdm 進度條
    pbar = tqdm(dataloader, desc="Training")
    
    for imgs, author_labels, style_labels in pbar:
        imgs = imgs.to(device)
        author_labels = author_labels.to(device)
        style_labels = style_labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        author_pred, style_pred = model(imgs)
        
        # Loss
        loss = criterion(author_pred, style_pred, author_labels, style_labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
    return running_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct_author = 0
    correct_style = 0
    total = 0
    
    with torch.no_grad():
        for imgs, author_labels, style_labels in tqdm(dataloader, desc="Validating"):
            imgs = imgs.to(device)
            author_labels = author_labels.to(device)
            style_labels = style_labels.to(device)
            
            author_pred, style_pred = model(imgs)
            
            loss = criterion(author_pred, style_pred, author_labels, style_labels)
            val_loss += loss.item()
            
            # 計算準確率
            _, pred_author = torch.max(author_pred, 1)
            _, pred_style = torch.max(style_pred, 1)
            
            correct_author += (pred_author == author_labels).sum().item()
            correct_style += (pred_style == style_labels).sum().item()
            total += author_labels.size(0)
            
    avg_loss = val_loss / len(dataloader)
    acc_author = 100 * correct_author / total
    acc_style = 100 * correct_style / total
    
    return avg_loss, acc_author, acc_style