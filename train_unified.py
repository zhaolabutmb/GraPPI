import argparse
import yaml
from typing import Dict
import torch
from utils.Training_modules.common_utils import set_seed
from utils.SSL_modules.SSL_train_module import run_ssl_pretraining
from utils.finetune_modules.finetune_dG_module import run_dG_finetuning
from utils.finetune_modules.finetune_disc_binder_module import run_disc_binder_finetuning
from utils.finetune_modules.finetune_ddG_module import run_ddg_finetuning
from utils.finetune_modules.finetune_ml_dG_module import run_ml_dG_finetuning
from utils.finetune_modules.finetune_ml_ddG_module import run_ml_ddG_finetuning
from utils.finetune_modules.finetune_ml_disc_binder_module import run_ml_disc_binder_finetuning
from baseline_modules import run_baseline_test
from utils.finetune_modules.esm_baseline_modules import run_esm_baseline_test


# ============================================================================
# CONFIGURATION LOADER
# ============================================================================
def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Unified Training: SSL Pre-training + Fine-tuning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('-config', type=str, required=True,
                       help='Path to YAML configuration file')
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Set seed
    set_seed(config['system']['seed'])
    
    # Setup device
    device = torch.device(f"cuda:{config['system']['cuda_id']}" 
                         if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*80}")
    print("UNIFIED TRAINING PIPELINE")
    print(f"{'='*80}")
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Seed: {config['system']['seed']}")
    print(f"Phases: {', '.join(config['training_task'])}")
    
    # Run phases
    ssl_checkpoint_path = None
    
    if 'ssl' in config['training_task']:
        ssl_checkpoint_path = run_ssl_pretraining(config, device)
    
    if 'dg_reg' in config['training_task']:
        run_dG_finetuning(config, device, ssl_checkpoint_path)
    
    if 'disc_binder' in config['training_task']:
        run_disc_binder_finetuning(config, device, ssl_checkpoint_path)
    
    if 'mutation' in config['training_task']:
        run_ddg_finetuning(config, device, ssl_checkpoint_path)
    
    if 'baseline' in config['training_task']:
        run_baseline_test(config, device)
    
    if 'esm_baseline' in config['training_task']:
        run_esm_baseline_test(config, device)
    
    if 'ml_dg_reg' in config['training_task']:
        run_ml_dG_finetuning(config, device, ssl_checkpoint_path)
    
    if 'ml_mutation' in config['training_task']:
        run_ml_ddG_finetuning(config, device, ssl_checkpoint_path)
    
    if 'ml_disc_binder' in config['training_task']:
        run_ml_disc_binder_finetuning(config, device, ssl_checkpoint_path)
    
    print(f"\n{'='*80}")
    print("ALL TRAINING PHASES COMPLETE!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
