import torch
import torch.nn as nn

class MLCLoss(nn.Module):
    def __init__(self, eps=0.01, gamma=1):
        super(MLCLoss, self).__init__()
        self.eps = eps
        self.gamma = gamma
        
    def compute_discrimn_loss(self, W):
        p, m = W.shape
        I = torch.eye(p,device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.
    
    def compute_compress_loss(self, W, Pi):
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p,device=W.device).expand((k,p,p))
        trPi = Pi.sum(2) + 1e-8
        scale = (p/(trPi*self.eps)).view(k,1,1)
        
        W = W.view((1,p,m))
        log_det = torch.logdet(I + scale*W.mul(Pi).matmul(W.transpose(1,2)))
        compress_loss = (trPi.squeeze()*log_det/(2*m)).sum()
        return compress_loss
        
    def forward(self, X, Y, detach=True):
        Pi = Y.T.reshape((Y.shape[0],1,-1))
        W = X.T
        discrimn_loss = self.compute_discrimn_loss(W)
        compress_loss = self.compute_compress_loss(W, Pi)
        total_loss = - discrimn_loss + self.gamma*compress_loss
        if detach:
            return total_loss, [discrimn_loss.item(), compress_loss.item()]
        else:
            return total_loss, [discrimn_loss, compress_loss]



class TotalCodingRate(nn.Module):
    def __init__(self, eps=0.01):
        super(TotalCodingRate, self).__init__()
        self.eps = eps
        
    def compute_discrimn_loss(self, W):
        """Discriminative Loss."""
        p, m = W.shape  #[d, B]
        I = torch.eye(p,device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.
    
    def forward(self,X):
        return - self.compute_discrimn_loss(X.T)