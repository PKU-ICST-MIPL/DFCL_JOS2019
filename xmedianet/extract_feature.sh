echo "extracting feature..."
th extract_feature.lua \
  -data_dir /home/zhuoyunkan/MM2018/data/data-XMN/ \
  -save_dir extracted_feature_iter40000/ \
  -gpuid 0 \
  -model ./models/xmedianet_40000.t7 
  
th extract_feature.lua \
  -data_dir /home/zhuoyunkan/MM2018/data/data-XMN/ \
  -save_dir extracted_feature_iter60000/ \
  -gpuid 0 \
  -model ./models/xmedianet_60000.t7 
