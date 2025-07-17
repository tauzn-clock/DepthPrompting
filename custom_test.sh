python3 custom_test.py  \
    --dir_data /scratchdata/nyu_plane \
    --model_name depth_prompt_main \
    --pretrain /scratchdata/pretrained/OURS/Depthprompting_depthformer_nyu.tar \
    --prop_kernel 9 \
    --conf_prop \
    --prop_time 18 \
    --nyu_val_samples 500 \
    --init_scailing \
    --gpu 0