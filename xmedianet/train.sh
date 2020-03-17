
th train.lua \
  -data_dir /home/zhuoyunkan/MM2018/data/data-XMN/ \
  -batch_size 20 \
  -learning_rate 0.00010 \
  -symmetric 1 \
  -max_epochs 400 \
  -savefile xmedianet \
  -num_caption 1 \
  -gpuid 1 \
  -print_every 1 \
  -nclass 200 \
  -img_dim 512 \
  -emb_dim 300 \
  -learning_rate_decay 1 \
  -save_every 5000 \
  -mmdweight 0.001 \
  -doc_length 65 \
  -checkpoint_dir models-0.001/ 
