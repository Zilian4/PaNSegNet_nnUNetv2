# slaug_torch.py
import torch
from bezier import intensity_remap as gla_augment
from lla import lla_augment
from saliency import compute_saliency, smooth_saliency
from fusion import sbf_fuse

class SLAugTorch:
    """
    Minimal SLAug pipeline:
      GLA -> LLA -> Saliency(Gla) -> SBF fuse
    """
    def __init__(
        self,
        saliency_model,
        classes,
        p: float = 1,
        num_knots: int = 8,
        strength_gla: float = 0.5,
        strength_lla: float = 0.5,
        sal_loss_type: str = "logit",      # start with "logit" for stability
        grid_size: int = 8,
        fusion_mode: str = "soft",
        fusion_threshold: float = 0.5,
        fusion_temperature: float = 5.0,
        apply_to_background: bool = False,
    ):
        self.saliency_model = saliency_model
        self.classes = classes
        self.p = p
        self.num_knots = num_knots
        self.strength_gla = strength_gla
        self.strength_lla = strength_lla
        self.sal_loss_type = sal_loss_type
        self.grid_size = grid_size
        self.fusion_mode = fusion_mode
        self.fusion_threshold = fusion_threshold
        self.fusion_temperature = fusion_temperature
        self.apply_to_background = apply_to_background

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        if (not self.p) or (torch.rand(1, device=x.device).item() > self.p):
            return x

        # 1) GLA
        x_gla = gla_augment(x, num_knots=self.num_knots, strength=self.strength_gla)

        # 2) LLA
        x_lla = lla_augment(
            x, y,
            classes=self.classes,
            num_knots=self.num_knots,
            strength=self.strength_lla,
            apply_to_background=self.apply_to_background
        )

        # 3) Saliency on GLA
        sal = compute_saliency(
            model=self.saliency_model,
            x=x_gla,
            y=y,
            loss_type=self.sal_loss_type,
            channel_reduce="mean",
        )
        sal = smooth_saliency(sal, grid_size=self.grid_size)

        # 4) SBF fuse
        x_fused = sbf_fuse(
            x=x,             
            x_gla=x_gla,
            x_lla=x_lla,
            sal=sal,
            mode=self.fusion_mode,
            threshold=self.fusion_threshold,
            temperature=self.fusion_temperature
        )
        # Z-score intensity normalization
        x_fused = (x_fused - x_fused.mean()) / x_fused.std()
        return x_fused

def load_saliency_model_from_checkpoint(checkpoint_path: str, device=None):
    """
    从checkpoint加载模型作为saliency_model
    """
    import sys
    import os
    # 添加nnUNet根目录到路径
    # 从当前文件位置: .../nnUNet/nnunetv2/training/nnUNetTrainer/slaug/slaug_torch.py
    # 到nnUNet根目录: .../nnUNet/
    current_file = os.path.abspath(__file__)
    nnunet_root = os.path.abspath(os.path.join(current_file, '../../../../..'))
    if nnunet_root not in sys.path:
        sys.path.insert(0, nnunet_root)
    
    from batchgenerators.utilities.file_and_folder_operations import load_json, join
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
    import nnunetv2
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 获取模型目录（checkpoint所在目录的父目录）
    model_dir = os.path.dirname(checkpoint_path)
    
    # 加载plans和dataset.json
    plans_file = join(model_dir, 'plans.json')
    dataset_json_file = join(model_dir, 'dataset.json')
    
    if not os.path.exists(plans_file):
        # 尝试从结果目录获取
        result_dir = os.path.dirname(model_dir)
        plans_file = join(result_dir, 'plans.json')
        dataset_json_file = join(result_dir, 'dataset.json')
    
    plans = load_json(plans_file)
    dataset_json = load_json(dataset_json_file)
    plans_manager = PlansManager(plans)
    
    # 获取配置信息
    trainer_name = checkpoint['trainer_name']
    configuration_name = checkpoint['init_args']['configuration']
    configuration_manager = plans_manager.get_configuration(configuration_name)
    
    # 获取trainer类并构建网络
    trainer_class = recursive_find_python_class(
        join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
        trainer_name, 
        'nnunetv2.training.nnUNetTrainer'
    )
    
    if trainer_class is None:
        raise RuntimeError(f'Unable to locate trainer class {trainer_name}')
    
    # 确定输入通道数
    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
    num_output_channels = plans_manager.get_label_manager(dataset_json).num_segmentation_heads
    
    # 构建网络
    network = trainer_class.build_network_architecture(
        configuration_manager.network_arch_class_name,
        configuration_manager.network_arch_init_kwargs,
        configuration_manager.network_arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision=False
    )
    
    # 加载权重
    network_weights = checkpoint['network_weights']
    # 处理可能的module.前缀
    new_state_dict = {}
    for k, v in network_weights.items():
        key = k
        if key.startswith('module.'):
            key = key[7:]
        new_state_dict[key] = v
    
    network.load_state_dict(new_state_dict, strict=False)
    network.eval()
    network.to(device)
    
    return network


if __name__ == "__main__":
    import torch.nn as nn
    import nibabel as nib
    import numpy as np
    import os
    import sys
    # 添加nnUNet根目录到路径
    current_file = os.path.abspath(__file__)
    nnunet_root = os.path.abspath(os.path.join(current_file, '../../../../..'))
    if nnunet_root not in sys.path:
        sys.path.insert(0, nnunet_root)
    
    # 加载数据
    input_image = nib.load("/data2/pyq6817/nnUNetv2/nnUNet_raw/Dataset404_AbdCT/imagesTr/Case_00002_0000.nii.gz")
    input_image = input_image.get_fdata()
    print("input_image",input_image.shape)
    input_image = torch.from_numpy(input_image).float()
    input_image = input_image.unsqueeze(0)
    input_image = input_image.unsqueeze(0)
    # input_image = input_image.cuda()
    # z-score normalization
    input_image = (input_image - input_image.mean()) / input_image.std()
    # input_image = input_image.clamp(0.0, 1.0)

    input_mask = nib.load("/data2/pyq6817/nnUNetv2/nnUNet_raw/Dataset404_AbdCT/labelsTr/Case_00002.nii.gz")
    input_mask = input_mask.get_fdata()
    input_mask = torch.from_numpy(input_mask).float()
    input_mask = input_mask.unsqueeze(0)
    input_mask = input_mask.unsqueeze(0)
    print("input_image",input_image.shape)
    print("input_mask",input_mask.shape)
    
    # 加载saliency模型
    checkpoint_path = "/data2/pyq6817/nnUNetv2/nnUNet_results/Dataset404_AbdCT/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth"
    print(f"\n加载saliency模型: {checkpoint_path}")
    saliency_model = None
    try:
        saliency_model = load_saliency_model_from_checkpoint(checkpoint_path)
        print("✓ 模型加载成功")
    except Exception as e:
        import traceback
        print(f"✗ 模型加载失败: {e}")
        print("提示: 如果缺少依赖（如acvl_utils），请确保nnUNet环境已正确配置")
        print("使用简单的测试模型作为fallback")
        saliency_model = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.Conv3d(16, 16, kernel_size=3, padding=1), 
            nn.ReLU()
        )
    
    # 定义4组不同的增强参数（移除hard fusion）
    param_sets = [
        {
            "name": "aug_1_low_strength",
            "p": 1.0,
            "num_knots": 8,
            "strength_gla": 0.2,
            "strength_lla": 0.2,
            "grid_size": 8,
            "fusion_mode": "soft",
            "fusion_threshold": 0.5,
            "fusion_temperature": 2.0,
        },
        {
            "name": "aug_2_medium_strength",
            "p": 1.0,
            "num_knots": 8,
            "strength_gla": 0.4,
            "strength_lla": 0.4,
            "grid_size": 8,
            "fusion_mode": "soft",
            "fusion_threshold": 0.5,
            "fusion_temperature": 5.0,
        },
        {
            "name": "aug_3_high_strength",
            "p": 1.0,
            "num_knots": 8,
            "strength_gla": 0.6,
            "strength_lla": 0.6,
            "grid_size": 8,
            "fusion_mode": "soft",
            "fusion_threshold": 0.5,
            "fusion_temperature": 8.0,
        },
        {
            "name": "aug_4_fine_grid",
            "p": 1.0,
            "num_knots": 12,
            "strength_gla": 0.5,
            "strength_lla": 0.5,
            "grid_size": 4,
            "fusion_mode": "soft",
            "fusion_threshold": 0.5,
            "fusion_temperature": 5.0,
        },
    ]
    
    outputs = []
    print("\n" + "="*80)
    print(f"使用{len(param_sets)}组不同参数进行增强（仅soft模式）")
    print("="*80)
    
    for i, params in enumerate(param_sets, 1):
        print(f"\n[{i}/{len(param_sets)}] 参数组: {params['name']}")
        print(f"  strength_gla={params['strength_gla']}, strength_lla={params['strength_lla']}")
        print(f"  num_knots={params['num_knots']}, grid_size={params['grid_size']}")
        print(f"  fusion_mode={params['fusion_mode']}, temperature={params['fusion_temperature']}")
        
        slaug = SLAugTorch(
            saliency_model=saliency_model,
            classes=[1, 2, 3, 4, 5],
            p=params['p'],
            num_knots=params['num_knots'],
            strength_gla=params['strength_gla'],
            strength_lla=params['strength_lla'],
            grid_size=params['grid_size'],
            fusion_mode=params['fusion_mode'],
            fusion_threshold=params['fusion_threshold'],
            fusion_temperature=params['fusion_temperature'],
        )
        
        output = slaug(input_image, input_mask)
        output_np = output.squeeze(0).squeeze(0).cpu().numpy()
        outputs.append((params['name'], output, output_np))
        
        # 保存增强后的图片
        output_file = f"{params['name']}.nii.gz"
        nib.save(nib.Nifti1Image(output_np, np.eye(4)), output_file)
        print(f"  ✓ 已保存: {output_file}")
        print(f"  统计: 均值={output.mean().item():.6f}, 标准差={output.std().item():.6f}, "
              f"最小值={output.min().item():.6f}, 最大值={output.max().item():.6f}")
    
    # 对比分析
    print("\n" + "="*80)
    print(f"对比分析：检查{len(param_sets)}组增强结果是否不同")
    print("="*80)
    
    # 计算每对增强结果之间的差异
    print(f"\n增强结果之间的差异矩阵 (平均绝对差异):")
    print(" " * 20, end="")
    for j in range(len(param_sets)):
        print(f"  Aug{j+1:2d}", end="")
    print()
    
    for i in range(len(param_sets)):
        print(f"Aug{i+1:2d} ({outputs[i][0]:20s})", end="")
        for j in range(len(param_sets)):
            if i == j:
                print("  0.0000", end="")
            else:
                diff = torch.abs(outputs[i][1] - outputs[j][1]).mean().item()
                print(f"  {diff:.4f}", end="")
        print()
    
    # 详细对比
    print("\n详细对比:")
    for i in range(len(param_sets)):
        for j in range(i+1, len(param_sets)):
            diff = torch.abs(outputs[i][1] - outputs[j][1])
            mean_diff = diff.mean().item()
            max_diff = diff.max().item()
            std_diff = diff.std().item()
            diff_ratio = (diff > 1e-4).sum().item() / diff.numel() * 100
            
            print(f"\n{outputs[i][0]} vs {outputs[j][0]}:")
            print(f"  平均绝对差异: {mean_diff:.6f}")
            print(f"  最大绝对差异: {max_diff:.6f}")
            print(f"  差异标准差: {std_diff:.6f}")
            print(f"  有显著差异的体素比例: {diff_ratio:.2f}%")
            
            is_different = mean_diff > 1e-4
            if is_different:
                print(f"  ✓ 两组增强结果不同")
            else:
                print(f"  ✗ 警告：两组增强结果几乎相同")
    
    # 与原始图像对比
    print("\n" + "="*80)
    print("与原始图像对比")
    print("="*80)
    input_normalized = (input_image - input_image.min()) / (input_image.max() - input_image.min() + 1e-8)
    
    for name, output, _ in outputs:
        output_normalized = (output - output.min()) / (output.max() - output.min() + 1e-8)
        diff = torch.abs(output_normalized - input_normalized)
        mean_diff = diff.mean().item()
        max_diff = diff.max().item()
        diff_ratio = (diff > 1e-4).sum().item() / diff.numel() * 100
        
        print(f"\n{name} vs 原始图像:")
        print(f"  平均绝对差异: {mean_diff:.6f}")
        print(f"  最大绝对差异: {max_diff:.6f}")
        print(f"  有显著差异的体素比例: {diff_ratio:.2f}%")
    
    print("\n" + "="*80)
    print("所有增强图片已保存完成！")
    print("="*80 + "\n")
    