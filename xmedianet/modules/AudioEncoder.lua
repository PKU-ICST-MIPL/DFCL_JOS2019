require 'modules.lstm_level'

local AudioEncoder = {}
function AudioEncoder.cnn(audio_dim, emb_dim, dropout, avg, cnn_dim)
  dropout = dropout or 0.0
  avg = avg or 0
  cnn_dim = cnn_dim or 256

  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- Bag

  local net = nn.Sequential()
  net:add(nn.View(audio_dim):setNumInputDims(2))
  net:add(nn.Linear(audio_dim, cnn_dim))
  net:add(nn.View(20,cnn_dim):setNumInputDims(2))

  local h1 = net(inputs[1])
  local lstm = nn.lstm(cnn_dim, 2, 0.5, 20)
  local r2 = lstm(h1)
  out = nn.Linear(cnn_dim, emb_dim)(nn.Dropout(dropout)(nn.View(cnn_dim):setNumInputDims(2)(r2)))
  out = nn.View(20,emb_dim):setNumInputDims(2)(out)
  local outputs = {}
  table.insert(outputs, out)
  return nn.gModule(inputs, outputs)
end

return AudioEncoder

