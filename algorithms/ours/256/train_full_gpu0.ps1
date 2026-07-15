Set-Location $PSScriptRoot
$env:CUDA_VISIBLE_DEVICES = "0"
python train_full.py --config configs/full_shared_dun.yaml --gpu 0
