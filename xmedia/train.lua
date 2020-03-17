require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'hdf5'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a multi-modal embedding model')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','','data directory.')
cmd:option('-batch_size',40,'number of sequences to train on in parallel')

cmd:option('-img_length',49,'image part size')
cmd:option('-img_dim',512,'image feature dimension')
cmd:option('-aud_length',20,'audio part size')
cmd:option('-aud_dim',128,'audio feature dimension')
cmd:option('-vid_length',20,'video part size')
cmd:option('-vid_dim',4096,'video feature dimension')
cmd:option('-td_length',47,'3d part size')
cmd:option('-td_dim',100,'3d feature dimension')
cmd:option('-doc_length',632,'document length')
cmd:option('-emb_dim',1536,'embedding dimension')

cmd:option('-nclass',200,'number of classes')
cmd:option('-dropout',0.0,'dropout rate')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-savefile','xmedia','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-checkpoint_dir', 'models/', 'output directory where checkpoints get written')
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
cmd:option('-max_epochs',300,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-learning_rate',0.0004,'learning rate')
cmd:option('-learning_rate_decay',0.98,'learning rate decay')
cmd:option('-learning_rate_decay_after',1,'in number of epochs, when to start decaying the learning rate')
cmd:option('-print_every',100,'how many steps/minibatches between printing out the loss')
cmd:option('-save_every',1000,'every how many iterations should we evaluate on validation data?')
cmd:option('-symmetric',1,'whether to use symmetric form of SJE')
cmd:option('-num_caption',5,'number of captions per image to be used for training')
cmd:option('-avg', 0, 'whether to time-average hidden units')
cmd:option('-cnn_dim', 512, 'char-cnn embedding dimension')
cmd:option('-mmdweight', 0.01, '')
cmd:option('-wd', 0.01, '')
cmd:option('-lambda', 0.005, '')
cmd:option('-margin', 0.5, '')

opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
print(opt)

local FiveMediaNetwork = require('modules.FiveMediaNetwork')
local AttentionModel = require('modules.attention')
local ClassifyModel = require('modules.classify')
local dataLoader = require('util.dataLoader')
local model_utils = require('util.model_utils')
local MMDloss = require('modules.mmd')


-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

local loader = dataLoader.create(
    opt.data_dir, opt.nclass, 
    opt.img_length, opt.img_dim, 
    opt.aud_length, opt.aud_dim, 
    opt.vid_length, opt.vid_dim, 
    opt.td_length, opt.td_dim, 
    opt.doc_length, opt.batch_size)

if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end

local do_random_init = false
if string.len(opt.init_from) > 0 then
    print('loading from checkpoint ' .. opt.init_from)
    local checkpoint = torch.load(opt.init_from)
    protos = checkpoint.protos
else
    protos = {}
    protos.fiveNet = nn.FiveMediaNetwork(loader.w2v_size, opt.img_dim, opt.aud_dim, opt.vid_dim, opt.td_dim, opt.dropout, opt.avg, opt.emb_dim, opt.cnn_dim)
    protos.img_attention = AttentionModel.atten(opt.emb_dim, opt.emb_dim, 49)
    protos.aud_attention = AttentionModel.atten(opt.emb_dim, opt.emb_dim, 20)
    protos.vid_attention = AttentionModel.atten(opt.emb_dim, opt.emb_dim, 20)
    protos.td_attention = AttentionModel.atten(opt.emb_dim, opt.emb_dim, 47)
    protos.txt_attention = AttentionModel.atten(opt.emb_dim, opt.emb_dim, 28)
    protos.img_classify = ClassifyModel.build(opt.emb_dim, opt.nclass)
    protos.aud_classify = ClassifyModel.build(opt.emb_dim, opt.nclass)
    protos.vid_classify = ClassifyModel.build(opt.emb_dim, opt.nclass)
    protos.td_classify = ClassifyModel.build(opt.emb_dim, opt.nclass)
    protos.txt_classify = ClassifyModel.build(opt.emb_dim, opt.nclass)

    protos.img_classify = require('weight-init')(protos.img_classify, 'xavier')
    protos.aud_classify = require('weight-init')(protos.aud_classify, 'xavier')
    protos.vid_classify = require('weight-init')(protos.vid_classify, 'xavier')
    protos.td_classify = require('weight-init')(protos.td_classify, 'xavier')
    protos.txt_classify = require('weight-init')(protos.txt_classify, 'xavier')

    protos.fiveNet:training()

    protos.img_classify:training()
    protos.aud_classify:training()
    protos.vid_classify:training()
    protos.td_classify:training()
    protos.txt_classify:training()

    protos.img_attention:training()
    protos.aud_attention:training()
    protos.vid_attention:training()
    protos.td_attention:training()
    protos.txt_attention:training()

    do_random_init = true

end

if opt.gpuid >= 0 then
    for k,v in pairs(protos) do
        if v.weights ~= nil then
            v.weights = v.weights:float():cuda()
            v.grads = v.grads:float():cuda()
        else
            v:cuda()
        end
    end
end
params, grad_params = model_utils.combine_all_parameters(protos.fiveNet, protos.img_classify, protos.txt_classify, protos.aud_classify, protos.vid_classify, protos.td_classify, protos.img_attention, protos.txt_attention,protos.aud_attention, protos.vid_attention, protos.td_attention)

-- reading category label and word2vec
local f = hdf5.open(path.join(opt.data_dir, 'category.hdf5'), 'r')
local w2v = f:read('w2v'):all()
local lookup = nn.LookupTable(21, 300)
lookup.weight:copy(w2v)
lookup.weight[1]:zero()
local category_lab = f:read('train'):all()
category_lab = category_lab[{1,{5,24}}]
local category = lookup:forward(category_lab)
category = nn.Normalize(2):forward(category)
category = category:cuda()

acc_batch = 0.0
acc_smooth = 0.0

-- loss function
function JointEmbeddingLoss(fea_txt, fea_img, fea_aud, fea_vid, fea_td, labels)
    local batch_size = fea_img:size(1)
    local num_class = loader.nclass
    local score = torch.zeros(batch_size, batch_size)
    local txt_grads = fea_txt:clone():fill(0)
    local img_grads = fea_img:clone():fill(0)
    local aud_grads = fea_aud:clone():fill(0)
    local vid_grads = fea_vid:clone():fill(0)
    local td_grads = fea_td:clone():fill(0)

    local loss = 0
    acc_batch = 0.0
    local margin = opt.margin
    for i = 1,batch_size do
	--txt
	local txt_score_sim = torch.dot(fea_txt:narrow(1,i,1), category:narrow(1,labels[i]+1,1))
	local tmp = torch.ceil(torch.rand(1) * opt.nclass)
	while (tmp[1] == labels[i]+1)
        do
            tmp = torch.ceil(torch.rand(1) * opt.nclass)
        end
	local txt_score_dsim = torch.dot(fea_txt:narrow(1,i,1), category:narrow(1,tmp[1],1))
	local thresh = txt_score_dsim - txt_score_sim + margin
	if (thresh > 0) then
            loss = loss + thresh
            txt_grads:narrow(1, i, 1):add(category:narrow(1,tmp[1],1)-category:narrow(1,labels[i]+1,1))
        end

	--img
	local img_score_sim = torch.dot(fea_img:narrow(1,i,1), category:narrow(1,labels[i]+1,1))
	local tmp = torch.ceil(torch.rand(1) * opt.nclass)
	while (tmp[1] == labels[i]+1)
        do
            tmp = torch.ceil(torch.rand(1) * opt.nclass)
        end
	local img_score_dsim = torch.dot(fea_img:narrow(1,i,1), category:narrow(1,tmp[1],1))
	local thresh = img_score_dsim - img_score_sim + margin
	if (thresh > 0) then
            loss = loss + thresh
            img_grads:narrow(1, i, 1):add(category:narrow(1,tmp[1],1)-category:narrow(1,labels[i]+1,1))
        end

	--aud
	local aud_score_sim = torch.dot(fea_aud:narrow(1,i,1), category:narrow(1,labels[i]+1,1))
	local tmp = torch.ceil(torch.rand(1) * opt.nclass)
	while (tmp[1] == labels[i]+1)
        do
            tmp = torch.ceil(torch.rand(1) * opt.nclass)
        end
	local aud_score_dsim = torch.dot(fea_aud:narrow(1,i,1), category:narrow(1,tmp[1],1))
	local thresh = aud_score_dsim - aud_score_sim + margin
	if (thresh > 0) then
            loss = loss + thresh
            aud_grads:narrow(1, i, 1):add(category:narrow(1,tmp[1],1)-category:narrow(1,labels[i]+1,1))
        end

	--vid
	local vid_score_sim = torch.dot(fea_vid:narrow(1,i,1), category:narrow(1,labels[i]+1,1))
	local tmp = torch.ceil(torch.rand(1) * opt.nclass)
	while (tmp[1] == labels[i]+1)
        do
            tmp = torch.ceil(torch.rand(1) * opt.nclass)
        end
	local vid_score_dsim = torch.dot(fea_vid:narrow(1,i,1), category:narrow(1,tmp[1],1))
	local thresh = vid_score_dsim - vid_score_sim + margin
	if (thresh > 0) then
            loss = loss + thresh
            vid_grads:narrow(1, i, 1):add(category:narrow(1,tmp[1],1)-category:narrow(1,labels[i]+1,1))
        end

	--td
	local td_score_sim = torch.dot(fea_td:narrow(1,i,1), category:narrow(1,labels[i]+1,1))
	local tmp = torch.ceil(torch.rand(1) * opt.nclass)
	while (tmp[1] == labels[i]+1)
        do
            tmp = torch.ceil(torch.rand(1) * opt.nclass)
        end
	local td_score_dsim = torch.dot(fea_td:narrow(1,i,1), category:narrow(1,tmp[1],1))
	local thresh = td_score_dsim - td_score_sim + margin
	if (thresh > 0) then
            loss = loss + thresh
            td_grads:narrow(1, i, 1):add((category:narrow(1,tmp[1],1)-category:narrow(1,labels[i]+1,1)))
        end
    end
    local denom = batch_size
    local res = { [1] = txt_grads:div(denom),
                  [2] = img_grads:div(denom),
		  [3] = aud_grads:div(denom),
		  [4] = vid_grads:div(denom),
		  [5] = td_grads:div(denom)}
    return loss / denom, res
end


function MMDLoss(fea_txt, fea_img, fea_aud, fea_vid, fea_td, labels, MMD_Criterion)
    local batch_size = fea_img:size(1)
    local num_class = loader.nclass
    local score = torch.zeros(batch_size, batch_size)
    local txt_grads = fea_txt:clone():fill(0)
    local img_grads = fea_img:clone():fill(0)
    local aud_grads = fea_aud:clone():fill(0)
    local vid_grads = fea_vid:clone():fill(0)
    local td_grads = fea_td:clone():fill(0)
    
    local loss = 0
    local tmp = MMD_Criterion:forward({fea_txt, fea_img})
    local tmp_grads = MMD_Criterion:backward({fea_txt, fea_img})
    loss = loss+tmp
    txt_grads:add(tmp_grads[1])
    img_grads:add(tmp_grads[2])
    local tmp = MMD_Criterion:forward({fea_txt, fea_aud})
    local tmp_grads = MMD_Criterion:backward({fea_txt, fea_aud})
    loss = loss+tmp
    txt_grads:add(tmp_grads[1])
    aud_grads:add(tmp_grads[2])
    local tmp = MMD_Criterion:forward({fea_txt, fea_vid})
    local tmp_grads = MMD_Criterion:backward({fea_txt, fea_vid})
    loss = loss+tmp
    txt_grads:add(tmp_grads[1])
    vid_grads:add(tmp_grads[2])
    local tmp = MMD_Criterion:forward({fea_txt, fea_td})
    local tmp_grads = MMD_Criterion:backward({fea_txt, fea_td})
    loss = loss+tmp
    txt_grads:add(tmp_grads[1])
    td_grads:add(tmp_grads[2])
    local tmp = MMD_Criterion:forward({fea_img, fea_aud})
    local tmp_grads = MMD_Criterion:backward({fea_img, fea_aud})
    loss = loss+tmp
    img_grads:add(tmp_grads[1])
    aud_grads:add(tmp_grads[2])
    local tmp = MMD_Criterion:forward({fea_img, fea_vid})
    local tmp_grads = MMD_Criterion:backward({fea_img, fea_vid})
    loss = loss+tmp
    img_grads:add(tmp_grads[1])
    vid_grads:add(tmp_grads[2])
    local tmp = MMD_Criterion:forward({fea_img, fea_td})
    local tmp_grads = MMD_Criterion:backward({fea_img, fea_td})
    loss = loss+tmp
    img_grads:add(tmp_grads[1])
    td_grads:add(tmp_grads[2])
    local tmp = MMD_Criterion:forward({fea_aud, fea_vid})
    local tmp_grads = MMD_Criterion:backward({fea_aud, fea_vid})
    loss = loss+tmp
    aud_grads:add(tmp_grads[1])
    vid_grads:add(tmp_grads[2])
    local tmp = MMD_Criterion:forward({fea_aud, fea_td})
    local tmp_grads = MMD_Criterion:backward({fea_aud, fea_td})
    loss = loss+tmp
    aud_grads:add(tmp_grads[1])
    td_grads:add(tmp_grads[2])
    local tmp = MMD_Criterion:forward({fea_vid, fea_td})
    local tmp_grads = MMD_Criterion:backward({fea_vid, fea_td})
    loss = loss+tmp
    vid_grads:add(tmp_grads[1])
    td_grads:add(tmp_grads[2])

    local res = { [1] = txt_grads,
                  [2] = img_grads,
          [3] = aud_grads,
          [4] = vid_grads,
          [5] = td_grads}
    return loss, res
end


-- check embedding gradient.
function wrap_emb(inp, nh, nx, ny, labs)
    local x = inp:narrow(1,1,nh*nx):clone():reshape(nx,nh)
    local y = inp:narrow(1,nh*nx + 1,nh*ny):clone():reshape(ny,nh)
    local loss, grads = JointEmbeddingLoss(x, y, labs)
    local dx = grads[1]
    local dy = grads[2]
    local grad = torch.cat(dx:reshape(nh*nx), dy:reshape(nh*ny))
    return loss, grad
end
if opt.checkgrad == 1 then
    print('\nChecking embedding gradient\n')
    local nh = 3
    local nx = 4
    local ny = 2
    local txt = torch.randn(nx, nh)
    local img = torch.randn(ny, nh)
    local labs = torch.randperm(nx)
    local initpars = torch.cat(txt:clone():reshape(nh*nx), img:clone():reshape(nh*ny))
    local opfunc = function(curpars) return wrap_emb(curpars, nh, nx, ny, labs) end
    diff, dC, dC_est = checkgrad(opfunc, initpars, 1e-3)
    print(dC)
    print(dC_est)
    print(diff)
    debug.debug()
end

local Criterion = nn.ClassNLLCriterion()
Criterion = Criterion:cuda()

local MMD_Criterion = nn.mmdCriterion()
MMD_Criterion = MMD_Criterion:cuda()

function feval_wrap(pars)
    ------------------ get minibatch -------------------
    local txt, img, aud, vid, td, labels = loader:next_batch()
    return feval(pars, txt, img, aud, vid, td, labels)
end

psi = torch.eye(5):div(5)

function feval(newpars, txt, img, aud, vid, td, labels)
    if newpars ~= params then
        params:copy(newpars)
    end
    grad_params:zero()

    if opt.gpuid >= 0 then -- ship the input arrays to GPU
        txt = txt:float():cuda()
        img = img:float():cuda()
        aud = aud:float():cuda()
        vid = vid:float():cuda()
        td = td:float():cuda()
        labels = labels:float():cuda()
    end
    ------------------- forward pass -------------------
    local fea_txt, fea_img, fea_aud, fea_vid, fea_td = protos.fiveNet:forward({txt, img, aud, vid, td})

    local txt_atten = protos.txt_attention:forward(fea_txt)
    local img_atten = protos.img_attention:forward(fea_img)
    local aud_atten = protos.aud_attention:forward(fea_aud)
    local vid_atten = protos.vid_attention:forward(fea_vid)
    local td_atten = protos.td_attention:forward(fea_td)

    -- Criterion --
    local loss, grads = JointEmbeddingLoss(txt_atten, img_atten, aud_atten, vid_atten, td_atten, labels)
    local dtxt = grads[1]       -- backprop through document CNN.
    local dimg = grads[2]       -- backprop through image encoder.
    local daud = grads[3]       -- backprop through image encoder.
    local dvid = grads[4]       -- backprop through image encoder.
    local dtd = grads[5]       -- backprop through image encoder.
    dis_loss = loss
    
    local loss2, grads2 = MMDLoss(txt_atten, img_atten, aud_atten, vid_atten, td_atten, labels, MMD_Criterion)
    local dtxt3 = grads2[1]       -- backprop through document CNN.
    local dimg3 = grads2[2]       -- backprop through image encoder.
    local daud3 = grads2[3]       -- backprop through image encoder.
    local dvid3 = grads2[4]       -- backprop through image encoder.
    local dtd3 = grads2[5]       -- backprop through image encoder.
    dis_loss2 = loss2

    local txt_cls = protos.txt_classify:forward(txt_atten)
    local img_cls = protos.img_classify:forward(img_atten)
    local aud_cls = protos.aud_classify:forward(aud_atten)
    local vid_cls = protos.vid_classify:forward(vid_atten)
    local td_cls = protos.td_classify:forward(td_atten)

    labels = labels+1
    err_txt = Criterion:forward(txt_cls, labels)
    local dtxt_cls = Criterion:backward(txt_cls, labels)
    local dtxt2 = protos.txt_classify:backward(txt_atten, dtxt_cls)

    err_img = Criterion:forward(img_cls, labels)
    local dimg_cls = Criterion:backward(img_cls, labels)
    local dimg2 = protos.img_classify:backward(img_atten, dimg_cls)

    err_aud = Criterion:forward(aud_cls, labels)
    local daud_cls = Criterion:backward(aud_cls, labels)
    local daud2 = protos.aud_classify:backward(aud_atten, daud_cls)

    err_vid = Criterion:forward(vid_cls, labels)
    local dvid_cls = Criterion:backward(vid_cls, labels)
    local dvid2 = protos.vid_classify:backward(vid_atten, dvid_cls)

    err_td = Criterion:forward(td_cls, labels)
    local dtd_cls = Criterion:backward(td_cls, labels)
    local dtd2 = protos.td_classify:backward(td_atten, dtd_cls)

    local txt_atten_grad = protos.txt_attention:backward(fea_txt, dtxt+dtxt2+dtxt3*opt.mmdweight)
    local img_atten_grad = protos.img_attention:backward(fea_img, dimg+dimg2+dimg3*opt.mmdweight)
    local aud_atten_grad = protos.aud_attention:backward(fea_aud, daud+daud2+daud3*opt.mmdweight)
    local vid_atten_grad = protos.vid_attention:backward(fea_vid, dvid+dvid2+dvid3*opt.mmdweight)
    local td_atten_grad = protos.td_attention:backward(fea_td, dtd+dtd2+dtd3*opt.mmdweight)
    
    protos.fiveNet:backward({txt, img, aud, vid, td}, {txt_atten_grad,img_atten_grad,aud_atten_grad,vid_atten_grad,td_atten_grad})

    return loss, grad_params
end

function getFlattenParameters(net)
    local p = net:parameters()
    local size = p[1]:nElement() + p[2]:nElement()
    local fp = torch.Tensor(size):fill(0)
    fp[{{1,p[1]:nElement()}}]:copy(p[1])
    fp[{{p[1]:nElement()+1,p[1]:nElement()+p[2]:nElement()}}]:copy(p[2])
    return fp
end

function getFlattenGradParameters(net)
    local _, p = net:parameters()
    local size = p[1]:nElement() + p[2]:nElement()
    local fp = torch.Tensor(size):fill(0)
    fp[{{1,p[1]:nElement()}}]:copy(p[1])
    fp[{{p[1]:nElement()+1,p[1]:nElement()+p[2]:nElement()}}]:copy(p[2])
    return fp
end

function combine_intermedia_parameters(...)
    local networks = {...}
    local pTemp = networks[1]:parameters()
    local size = pTemp[1]:nElement() + pTemp[2]:nElement()
    local parametersMatrix = torch.Tensor(size, #networks)
    for i = 1, #networks do
        local flatParams = getFlattenParameters(networks[i])
        parametersMatrix:narrow(2,i,1):copy(flatParams)
    end
    return parametersMatrix
end

-- start optimization here
train_losses = {}
val_losses = {}
local optim_state = {learningRate = opt.learning_rate, weightDecay = opt.wd}
local iterations = opt.max_epochs * loader.ntrain
local iterations_per_epoch = loader.ntrain
local loss0 = nil
for i = 1, iterations do
    local epoch = i / loader.ntrain

    local timer = torch.Timer()
    local _, loss = optim.rmsprop(feval_wrap, params, optim_state)
    local time = timer:time().real

    local train_loss = loss[1] -- the loss is inside a list, pop it
    train_losses[i] = train_loss

    -- exponential learning rate decay
    if i % opt.save_every == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
        end
    end

    -- every now and then or on last iteration
    if i % opt.save_every == 0 or i == iterations then
        -- evaluate loss on validation data
        local val_loss = 0
        val_losses[i] = val_loss

      	local savefile  = string.format('%s/%s_%d.t7', opt.checkpoint_dir, opt.savefile, i)
        print('saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.protos = protos
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.i = i
        checkpoint.epoch = epoch
        checkpoint.vocab = loader.vocab_mapping
        torch.save(savefile, checkpoint)
    end

    if i % opt.print_every == 0 then
        print(string.format("%d/%d (ep %.3f), loss=%4.2f, txt=%4.2f, img=%4.2f, aud=%4.2f, vid=%4.2f, 3d=%4.2f, t/b=%.2fs",
              i, iterations, epoch, dis_loss, err_txt, err_img, err_aud, err_vid, err_td, time))
    end

    if i % 10 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    if loss0 == nil then loss0 = loss[1] end

end

