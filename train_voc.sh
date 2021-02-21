# CUDA_VISIBLE_DEVICES=0,1,2,3 \
python train.py \
	--backbone resnet \
	--lr 0.007 \
	--workers 4 \
	--epochs 2 \
	--batch-size 5 \
	--gpu-ids 0 \
	--checkname deeplab-resnet \
	--eval-interval 1 \
	--dataset pascal \
	--workers 0 \
	--no-cuda
