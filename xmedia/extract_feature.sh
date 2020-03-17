echo "extracting feature..."
th extract_feature.lua \
  -data_dir /home/zhuoyunkan/MM2018/data/data-xmedia/ \
  -gpuid 3 \
  -model ./models/xmedia_8000.t7 
