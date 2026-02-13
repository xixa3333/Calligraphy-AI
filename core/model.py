import torch
import torch.nn as nn

class MultiTaskCNN(nn.Module):
    def __init__(self, num_authors, num_styles):
        super(MultiTaskCNN, self).__init__()
        
        # --- [優化 1 & 2] 加入 Batch Normalization ---
        # 順序通常是: Conv -> BN -> ReLU -> MaxPool
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), # 加入 BN
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), # 加入 BN
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Layer 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), # 加入 BN
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Layer 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), # 加入 BN
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.flatten_dim = 256 * 8 * 8
        
        # Branch Author
        self.author_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 512),
            nn.BatchNorm1d(512), # 全連接層也可以加 BN
            nn.ReLU(),
            nn.Dropout(0.5),     # [維持] Dropout 很重要
            nn.Linear(512, num_authors)
        )
        
        # Branch Style
        self.style_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_styles)
        )

        # --- [優化 3] 權重初始化 (He Initialization) ---
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming (He) Initialization
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feat = self.features(x)
        return self.author_fc(feat), self.style_fc(feat)

# --- [關鍵修改] 支援權重的損失函數 ---
class MultiTaskLoss(nn.Module):
    def __init__(self, author_weights=None, style_weights=None, device='cpu'):
        super(MultiTaskLoss, self).__init__()
        
        # Branch C (Author): 針對資料量不平衡，傳入 class_weights
        if author_weights is not None:
            self.author_criterion = nn.CrossEntropyLoss(weight=author_weights.to(device))
        else:
            self.author_criterion = nn.CrossEntropyLoss()
            
        # Branch B (Style): 同樣可以針對「行草」過少設定權重
        if style_weights is not None:
            self.style_criterion = nn.CrossEntropyLoss(weight=style_weights.to(device))
        else:
            self.style_criterion = nn.CrossEntropyLoss()

    def forward(self, author_pred, style_pred, author_target, style_target):
        # 計算兩個分支的 Loss
        loss_author = self.author_criterion(author_pred, author_target)
        loss_style = self.style_criterion(style_pred, style_target)
        
        # 總 Loss = 兩者相加 (也可以在此調整兩者比例，例如讓作者辨識佔 1.5 倍)
        return loss_author + loss_style