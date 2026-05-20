#!/bin/bash
source /data/dust/group/atlas/ttreco/venv/bin/activate

python scripts/HyperParameters.py --load_config config/load_test_config.yaml --evaluation_config config/hyperparams_regression.yaml --output_dir thesis_plots/hyperparams_regression --k_fold 5 
python scripts/HyperParameters.py --load_config config/load_test_config.yaml --evaluation_config config/hyperparams_assignment.yaml --output_dir thesis_plots/hyperparams_assignment --k_fold 5
python scripts/HyperParameters.py --load_config config/load_test_config.yaml --evaluation_config config/hyperparams_full_reco_ap_grad.yaml --output_dir thesis_plots/hyperparams_full_reco_ap_grad --k_fold 5
python scripts/HyperParameters.py --load_config config/load_test_config.yaml --evaluation_config config/hyperparams_full_reco.yaml --output_dir thesis_plots/hyperparams_full_reco --k_fold 5
python scripts/HyperParameters.py --load_config config/load_test_config.yaml --evaluation_config config/hyperparams_full_reco_baseline.yaml --output_dir thesis_plots/hyperparams_full_reco_baseline --k_fold 5
