GPU=0
CUDA_VISIBLE_DEVICES=${GPU} \
python main.py \
	--train_iam ../lmdb_formater/trainset/ \
	--valroot ../lmdb_formater/testset/ \
	--workers 3 \
	--batchSize 64 \
	--niter 70 \
	--lr 1 \
	--cuda \
	--experiment output/ \
	--displayInterval 100 \
	--valInterval 1000 \
	--saveInterval 40000 \
	--adadelta \
	--BidirDecoder