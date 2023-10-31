import torch
from torch import nn
from collections import OrderedDict
# from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


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
            seq_length = t ## 
            position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)## [0...seq_length]
            position_ids = position_ids.unsqueeze(0).expand(x.size(0), -1)## [0...seq_length]*b
            frame_position_embeddings = self.frame_position_embeddings(position_ids)## b*n_seq*dim  
            x = x + frame_position_embeddings## b*t*embed

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = x.type(x_original.dtype) #+ x_original

        else:
            raise ValueError('Unknown temporal modeling header: {}'.format(self.vid_header))
        x=torch.mean(x,dim=1)
        return x

def get_orthogonal_num_vec(now_tensor,best_tensor):
    l2sums=torch.norm(best_tensor,dim=1)**2
    l2sums=torch.unsqueeze(l2sums,dim=1)
    maps=(now_tensor*best_tensor)/l2sums
    maps*=best_tensor 
    gain=now_tensor-maps
    res=best_tensor+gain
    return res

class OTI(nn.Module):
    def __init__(self, clip_model, video_header, n_seg) :
        super(OTI, self).__init__()
        self.visual = clip_model.visual## visual encoder from CLIP
        self.fusion_model = video_header
        self.n_seg = n_seg # 8
        self.logit_scale = clip_model.logit_scale#tensor(4.6052, requires_grad=True)
        self.ratio=torch.nn.Parameter(torch.randint(5,6,(1,))/10) #torch.randint(5,6,(1,))/10
#         self.ratio.requires_grad=True
        self.all=torch.randint(1,2,(1,))
        
        

    def forward(self, images, text_emb):
        '''

        '''
        video_emb,video_emb_front = self.encoder_video(images)## b*dim
        video_emb = video_emb / video_emb.norm(dim=-1, keepdim=True)
        
        video_emb_front=video_emb_front/video_emb_front.norm(dim=-1, keepdim=True)
        new_vector=get_orthogonal_num_vec(video_emb,video_emb_front)
        
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()# 100
        logits_new = logit_scale * new_vector @ text_emb.t()## 
        logits_after=logit_scale*video_emb@text_emb.t()
        
        return video_emb,video_emb_front,logits_new,logits_after
    
    def encoder_video(self,images):

        b, t, c, h, w = images.size()  ## b:batch_size t: the num of frames
        images = images.view(-1, c, h, w)
        image_embs = self.visual(images)
#         image_emb = self.fc(image_emb)  ## b*frames*512

        if t == 1:  # joint  self.n_seg=1
            return image_embs
        else:
            
            image_embs = image_embs.view(b, t, -1)
            image_embs_front = torch.mean(image_embs,dim=1)
            image_embs = self.fusion_model(image_embs)
            return image_embs,image_embs_front  ## b*embed
