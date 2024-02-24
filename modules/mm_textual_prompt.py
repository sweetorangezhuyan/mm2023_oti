import torch
from torch import nn
from collections import OrderedDict
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from .text_prompt import TextPromptLearner

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    '''
    时序残差块，带attention_mask
    '''
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0] # 结果为tuple，选择index 0的tensor

    def forward(self, x: torch.Tensor):
        '''
        input:n_seg*b*dim
        output:n_seg*b*dim
        n_reg 片段的个数
        b 一个片段多少张视频帧
        '''
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x



class TemporalTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks((x))


class video_header(nn.Module):
    def __init__(self, vid_head, clip_state_dict):
        super().__init__()
        self.vid_header = vid_head
        assert vid_head in ["None", "Transf"]

        if self.vid_header == "Transf":
            embed_dim = clip_state_dict["text_projection"].shape[1]## vitb-16: 512

            context_length = clip_state_dict["positional_embedding"].shape[0]## 77
            vocab_size = clip_state_dict["token_embedding.weight"].shape[0]##49408
            transformer_width = clip_state_dict["ln_final.weight"].shape[0]## 512
            transformer_heads = transformer_width // 64 ## 8

            transformer_layers = len(
                set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

            self.frame_position_embeddings = nn.Embedding(context_length, embed_dim)## 77*512

            self.transformer = TemporalTransformer(width=embed_dim, layers=1, heads=transformer_heads)
            print('layer=1')

        self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x):
        b, t, c = x.size()# b: step   t:n_seg
        x = x.contiguous()
        if self.vid_header == "None":
            pass

        elif self.vid_header == "Transf":
            x_original = x
            seq_length = t ## 片段的个数
            position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)## [0...seq_length]
            position_ids = position_ids.unsqueeze(0).expand(x.size(0), -1)## [0...seq_length]*b
            ## 片段内的所有帧图片共享同一个位置编码，以片段为基础进行位置编码
            frame_position_embeddings = self.frame_position_embeddings(position_ids)## b*n_seq*dim  每个b的内容是一样的
            x = x + frame_position_embeddings## b*t*embed

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = x.type(x_original.dtype) #+ x_original

        else:
            raise ValueError('Unknown temporal modeling header: {}'.format(self.vid_header))
        x=torch.mean(x,dim=1)#.values
#         x_original=torch.max(x_original,dim=1).values
        return x

def get_orthogonal_num_vec(now_tensor,best_tensor):
    l2sums=torch.norm(best_tensor,dim=1)**2
    l2sums=torch.unsqueeze(l2sums,dim=1)
    maps=(now_tensor*best_tensor)/l2sums
    maps*=best_tensor 
    gain=now_tensor-maps
    res=best_tensor+gain
    return res

'''
text_encoder 模型
'''
class Textencoder(nn.Module):
    def __init__(self,clip_model) :
        super().__init__()
        # self.vocab_size=clip_model.vocab_size
        self.token_embedding=clip_model.token_embedding
        self.transformer=clip_model.transformer
        self.positional_embedding=clip_model.positional_embedding
        self.emb_dropout = clip_model.emb_dropout
        self.text_projection =clip_model.text_projection
        self.ln_final=clip_model.ln_final
    def forward(self, prompts,tokenized_prompts):
        '''
        prompts: 为prompt+class_name token_embedding的结果
        tokenized_prompts：为prompt+class_name 分词的结果
        '''
        # x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = prompts + self.positional_embedding
        if self.emb_dropout > 0:
            x = self.dropout(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x 

class OTI_TPT(nn.Module):
    def __init__(self,clip_model, video_header,class_name, datas):
        super(OTI_TPT,self).__init__()
        '''
        视觉分支与textual prompt tuning
        '''
        self.visual = clip_model.visual## 视觉编码器 使用clip的visual encoder
        self.fusion_model = video_header
        self.n_seg = datas.num_segments # 8
        self.logit_scale = clip_model.logit_scale#tensor(4.6052, requires_grad=True)
        # 每次forward都要更新文本编码器部分，且文本部分的参数固定
        self.textual=Textencoder(clip_model=clip_model)
        # prompt 构建，传入参数
        self.prompt_learner=TextPromptLearner(
            classnames=class_name,
            text_model=self.textual,
            num_prompts= datas.num_prompts,
            prompt_init=datas.prompt_init,
            CSC=datas.CSC,
            ctx_pos=datas.ctx_pos
            )
        self.tokenized_prompts=self.prompt_learner.tokenized_prompts
        self._freeze_textual()


    def _freeze_textual(self):
        for name, param in self.textual.named_parameters():
            param.requires_grad = False


    def forward(self,images):
        # 视频特征
        images_emb,images_emb_front = self.encode_videos(images)## 一个clip的帧数 b*dim
        image_emb = images_emb / images_emb.norm(dim=-1, keepdim=True)
        image_emb_front=images_emb_front/images_emb_front.norm(dim=-1, keepdim=True)
        new_vector=get_orthogonal_num_vec(image_emb,image_emb_front)
        logit_scale = self.logit_scale.exp()# 100
        # 类别特征
        promptes=self.prompt_learner()
        tokenized_prompts=self.tokenized_prompts
        text_emb=self.textual(promptes,tokenized_prompts)
        '''
        为什么这里不加torch.no_grad()
        因为需要对text_emb的结果进行学习，对ctx进行学习
        如果requires_grad=False,grad_fn也为none时，不会进行求导，参数不会更新
        '''
        text_emb=text_emb/text_emb.norm(dim=-1,keepdim=True)
        
        #分类响应
        logits_new = logit_scale * new_vector @ text_emb.t()## 一个片段的视频帧数*分类个数
        logits_after=logit_scale*image_emb@text_emb.t()
        return image_emb,image_emb_front,logits_new,logits_after
    
    def encode_videos(self, video_frames):
        b, t, c, h, w = video_frames.size()  ## b:batch_size t:一条视频的视频帧数
        #         b = bt // self.n_seg## self.n_seg片段的个数,b表示每个片段多少帧图片
        video_frames = video_frames.view(-1, c, h, w)
        images_emb = self.visual(video_frames)
#         image_emb = self.fc(image_emb)  ## b*frames*512

        if t == 1:  # joint  self.n_seg=1
            return images_emb
        else:
            
            images_emb = images_emb.view(b, t, -1)
            images_emb_front = torch.mean(images_emb,dim=1)
            images_emb = self.fusion_model(images_emb)
            return images_emb,images_emb_front  ## b*embed


class VideoCLIP(nn.Module):
    def __init__(self, clip_model, video_header, n_seg) :
        super(VideoCLIP, self).__init__()
        self.visual = clip_model.visual## 视觉编码器 使用clip的visual encoder
        self.fusion_model = video_header
        self.n_seg = n_seg # 8
        self.logit_scale = clip_model.logit_scale#tensor(4.6052, requires_grad=True)
        self.ratio=torch.nn.Parameter(torch.randint(5,6,(1,))/10) #torch.randint(5,6,(1,))/10
#         self.ratio.requires_grad=True
        self.all=torch.randint(1,2,(1,))
        
        

    def forward(self, image, text_emb):
        '''

        '''
        image_emb,image_emb_front = self.encode_image(image)## 一个clip的帧数 b*dim
        image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
        
        image_emb_front=image_emb_front/image_emb_front.norm(dim=-1, keepdim=True)
        new_vector=get_orthogonal_num_vec(image_emb,image_emb_front)
#         self.ratio=self.ratio.to(image_emb.device)
#         self.all=self.all.to(image_emb.device)
# #         print(self.ratio.device,self.all.device)
#         image_end=self.ratio*image_emb+(self.all-self.ratio)*image_emb_front
        
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()# 100
        logits_new = logit_scale * new_vector @ text_emb.t()## 一个片段的视频帧数*分类个数
        logits_after=logit_scale*image_emb@text_emb.t()
        
        
        return image_emb,image_emb_front,logits_new,logits_after

    def encode_image(self, image):
        b, t, c, h, w = image.size()  ## b:batch_size t:一条视频的视频帧数
        #         b = bt // self.n_seg## self.n_seg片段的个数,b表示每个片段多少帧图片
        image = image.view(-1, c, h, w)
        image_emb = self.visual(image)
#         image_emb = self.fc(image_emb)  ## b*frames*512

        if t == 1:  # joint  self.n_seg=1
            return image_emb
        else:
            
            image_emb = image_emb.view(b, t, -1)
            image_emb_front = torch.mean(image_emb,dim=1)
            image_emb = self.fusion_model(image_emb)
            return image_emb,image_emb_front  ## b*embed
