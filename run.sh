CUDA_VISIBLE_DEVICES=4,5,6,7 python3 trainIcdar15_amp.py \
    --results_dir ./results_dir/exp_official_craft_supervision_shwang \
    --synthData_dir /data/SynthText/ \
    --icdar2015_dir /data/ICDAR2015/ \
    --test_folder /data/ICDAR2015/ \
    --end_iter 50000 \
    --batch_size 20 \
    --lr_decay 15000 \
    --gamma 0.2 \
    --loss 3 \
    --ckpt_path /nas/home/gmuffiness/workspace/ocr_related/daintlab-CRAFT-Reimplementation/clean-code/exp/shwang_synthtext_test6_26/CRAFT_clr_amp_60000.pth \
    --amp True \
    --aug True;


# --ckpt_path /nas/home/shwang/workspace/craft/exp/synthtext_test6_26/CRAFT_clr_50000.pth \