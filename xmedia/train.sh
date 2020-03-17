
th train.lua \
  -data_dir /home/zhuoyunkan/MM2018/data/data-xmedia/ \
  -batch_size 20 \
  -learning_rate 0.00010 \
  -symmetric 1 \
  -max_epochs 500 \
  -savefile xmedia \
  -num_caption 1 \
  -gpuid 0 \
  -print_every 1 \
  -nclass 20 \
  -img_dim 512 \
  -doc_length 28 \
  -emb_dim 300 \
  -learning_rate_decay 1 \
  -save_every 2000 \
  -mmdweight 0.001 \
  -checkpoint_dir models/
