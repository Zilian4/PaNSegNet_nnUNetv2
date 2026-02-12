import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss

class DomainAdaptationLoss(nn.Module):
    def __init__(self,seg_loss:nn.Module, domain_loss:nn.Module, lambda_da:float=0.1):
        super().__init__()
        self.seg_loss = seg_loss
        self.domain_loss = domain_loss
        self.lambda_da = lambda_da
    
    def forward(self, 
                seg_output: torch.Tensor, 
                seg_target: torch.Tensor,
                domain_output: torch.Tensor,
                domain_target: torch.Tensor) -> dict:
        seg_loss = self.seg_loss(seg_output, seg_target)
        domain_loss = self.domain_loss(domain_output, domain_target)
        total_loss = seg_loss*(1-self.lambda_da) + domain_loss*self.lambda_da
        return {
            'total_loss': total_loss,
            'segmentation_loss': seg_loss,
            'domain_loss': domain_loss,
        }

    def set_lambda_da(self, lambda_da: float):
        self.lambda_da = lambda_da  
        


if __name__ == "__main__":
    # Test the domain adaptation loss
    import torch
    
    # Create dummy data
    batch_size, num_classes, depth, height, width = 2, 4, 32, 32, 32
    seg_output = torch.randn(batch_size, num_classes, depth, height, width)
    seg_target = torch.randint(0, num_classes, (batch_size, 1, depth, height, width))
    
    domain_output = torch.randn(batch_size, 1, depth, height, width)
    domain_target = torch.randint(0, 2, (batch_size,))
    
    # Create loss functions
    seg_loss = DC_and_CE_loss(
        soft_dice_kwargs={'batch_dice': True},
        ce_kwargs={'ignore_index': None},
        weight_ce=1,
        weight_dice=1
    )
    
    domain_loss = nn.BCEWithLogitsLoss()
    
    # Test domain adaptation loss
    da_loss = DomainAdaptationLoss(
        seg_loss=seg_loss,
        domain_loss=domain_loss,
        lambda_da=1.0
    )
    
    loss_dict = da_loss(seg_output, seg_target, domain_output, domain_target)
    print("Domain Adaptation Loss:", loss_dict)
    
    # Test identifier-based domain target creation
    identifiers = ['case_001_domain0', 'case_002_domain0', 'case_003_domain1', 'case_004_domain1']
    domain_targets = create_domain_target_from_identifiers_simple(
        identifiers, 
        domain0_patterns=['domain0'], 
        domain1_patterns=['domain1']
    )
    print("Domain targets:", domain_targets) 