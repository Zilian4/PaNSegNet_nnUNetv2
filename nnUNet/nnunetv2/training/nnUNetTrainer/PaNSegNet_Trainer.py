from typing import Union, List, Tuple

import torch
from torch import nn

from nnunetv2.network_architecture.PaNSegNet import PaNSegNet
from .nnUNetTrainer import nnUNetTrainer


class PaNSegNet_Trainer(nnUNetTrainer):
    """
    Custom trainer for PaNSegNet architecture.
    This trainer uses the custom PaNSegNet network architecture instead of the default nnU-Net architectures.
    """
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        # Call parent class initialization
        super().__init__(plans, configuration, fold, dataset_json, device)

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        Build the PaNSegNet architecture according to the plans.
        
        This is the function that is called in inference as well! This is needed so that all network architecture
        variants can be loaded at inference time (inference will use the same nnUNetTrainer that was used for
        training, so if you change the network architecture during training by deriving a new trainer class then
        inference will know about it).

        If you need to know how many segmentation outputs your custom architecture needs to have, use the following snippet:
        > label_manager = plans_manager.get_label_manager(dataset_json)
        > label_manager.num_segmentation_heads
        (why so complicated? -> We can have either classical training (classes) or regions. If we have regions,
        the number of outputs is != the number of classes. Also there is the ignore label for which no output
        should be generated. label_manager takes care of all that for you.)

        """
        import pydoc
        
        # Convert string-based class references to actual classes
        architecture_kwargs = dict(**arch_init_kwargs)
        for ri in arch_init_kwargs_req_import:
            if architecture_kwargs.get(ri) is not None:
                architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])
        
        # Set deep supervision
        architecture_kwargs['deep_supervision'] = enable_deep_supervision
        
        # Build PaNSegNet with the parameters from plans
        # Note: PaNSegNet expects 'n_blocks_per_stage' but plans might have 'n_conv_per_stage'
        # We need to map the parameters correctly
        if 'n_conv_per_stage' in architecture_kwargs and 'n_blocks_per_stage' not in architecture_kwargs:
            architecture_kwargs['n_blocks_per_stage'] = architecture_kwargs.pop('n_conv_per_stage')
        
        # Create the network
        network = PaNSegNet(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            **architecture_kwargs
        )
        
        # Initialize weights if the network has an initialize method
        if hasattr(network, 'initialize'):
            network.apply(network.initialize)
        
        return network
