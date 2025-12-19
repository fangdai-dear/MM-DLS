import torch
import torch.nn as nn

class CoxPHLoss(nn.Module):
    """
    实现 Cox Proportional Hazards Loss (负对数偏似然)
    """
    def __init__(self):
        super(CoxPHLoss, self).__init__()

    def forward(self, risk_pred, durations, events):
        """
        risk_pred: [batch_size] 模型输出的风险评分（未经过sigmoid）
        durations: [batch_size] 存活时间
        events:    [batch_size] 事件发生标志 (1=死亡/复发, 0=删失)
        """
        # 以时间降序排序（从最长生存期开始）
        order = torch.argsort(durations, descending=True)
        risk_pred = risk_pred[order]
        events = events[order]

        # 累加风险值 log-sum-exp 以稳定训练
        log_cumsum = torch.logcumsumexp(risk_pred, dim=0)
        diff = risk_pred - log_cumsum
        loss = -torch.sum(diff * events) / torch.sum(events + 1e-8)  # 防止除以 0

        return loss
