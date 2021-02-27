# CUDA_VISIBLE_DEVICES=0,1,2,3 \
python train.py \
	--backbone xception \
	--in-channels 4 \
	--lr 0.007 \
	--workers 0 \
	--epochs 30 \
	--batch-size 16 \
	--gpu-ids 0,1 \
	--dataset suichang_round1\
	--do-eval \
	--eval-interval 1 \
	--test-dir /root/datasets/suichang_round1/suichang_round1_test_partA_210120 \
