import torch
import clip

import torch.nn as nn
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
def text_prompt(data):
    # text_aug = ['{}']
    text_aug = ['a video of a person {}.']

    # Kinetics
    # text_aug = [
    #     'a photo of a person {}.',
    #     'a photo of {}.',
    #     'a photo of a person using {}.',
    #     'a photo of a person doing {}.',
    #     'a photo of a person during {}.',
    #     'a photo of a person performing {}.',
    #     'a photo of a person practicing {}.',
    #     'a video of {}.',
    #     'a video of a person {}.',
    #     'a video of a person using {}.',
    #     'a video of a person doing {}.',
    #     'a video of a person during {}.',
    #     'a video of a person performing {}.',
    #     'a video of a person practicing {}.',
    #     'a example of {}.',
    #     'a example of a person {}.',
    #     'a example of a person using {}.',
    #     'a example of a person doing {}.',
    #     'a example of a person during {}.',
    #     'a example of a person performing {}.',
    #     'a example of a person practicing {}.',
    #     'a demonstration of {}.',
    #     'a demonstration of a person {}.',
    #     'a demonstration of a person using {}.',
    #     'a demonstration of a person doing {}.',
    #     'a demonstration of a person during {}.',
    #     'a demonstration of a person performing {}.',
    #     'a demonstration of a person practicing {}.',
    # ]

    text_dict = {}
    num_text_aug = len(text_aug)

    for ii, txt in enumerate(text_aug):
        text_dict[ii] = torch.cat([clip.tokenize(txt.format(c)) for  c in data.classes])

    classes = text_dict[0] 

    return classes, num_text_aug, text_dict

## tokenize 来自clip
## 返回的prompts 是token_embedding 的结果
class TextPromptLearner(nn.Module):
    def __init__(self,classnames, text_model, num_prompts, prompt_init='',CSC=False,ctx_pos='end'):
        super().__init__()
        _tokenizer=_Tokenizer() # 对类别名称进行编码
        n_cls=len(classnames)
        n_ctx=num_prompts
        ctx_init=prompt_init
        ctx_dim=text_model.ln_final.weight.shape[0]

        if ctx_init:
            ctx_init=ctx_init.replace('_',' ')
            n_ctx=len(ctx_init.split())
            prompt=clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding=text_model.token_embedding(prompt)
            ctx_vectors=embedding[0,1:1+n_ctx,:]
            prompt_prefix=ctx_init
        else:
            if CSC:
                print("Initializing class-specific contexts")
                ctx_vectors=torch.empty(n_cls,n_ctx,ctx_dim)
            else:
                print("Initializing a generic context")
                ctx_vectors=torch.empty(n_ctx,ctx_dim)
            nn.init.normal_(ctx_vectors,std=0.02)
            prompt_prefix=" ".join(["X"]*n_ctx)
        
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx=nn.Parameter(ctx_vectors)

        classnames=[name.replace('_',' ') for name in classnames]
        name_lens=[len(_tokenizer.encode(name)) for name in classnames]
        prompts=[prompt_prefix+' '+name+'.' for name in classnames]
        print(prompts[0])


        tokenized_prompts=torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding=text_model.token_embedding(tokenized_prompts)
        
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = ctx_pos
    def forward(self):
        ctx=self.ctx # prompt的向量
        if ctx.dim()==2:
            ctx=ctx.unsqueeze(0).expand(self.n_cls,-1,-1)

        prefix=self.token_prefix
        suffix=self.token_suffix

        if self.class_token_position=='end':
            prompts=torch.cat([
                prefix,ctx,suffix],dim=1)
            
        elif self.class_token_position=='middle':
            half_n_ctx=self.n_ctx//2
            prompts=[]
            for i in range(self.n_cls):
                name_len=self.name_lens[i]
                prefix_i=prefix[i:i+1,:,:]
                class_i=suffix[i:i+1,:name_len,:]
                suffix_i=suffix[i:i+1,name_len:,:]
                ctx_i_half1=ctx[i:i+1,:half_n_ctx,:]
                ctx_i_half2=ctx[i:i+1,half_n_ctx:,:]
                prompt=torch.cat([
                    prefix_i,
                    ctx_i_half1,
                    class_i,
                    ctx_i_half2,
                    suffix_i],dim=1)
                prompts.append(prompt)
            prompts=torch.cat(prompts,dim=0)
        elif self.class_token_position=='front':
            prompts=[]
            for i in range(self.n_cls):
                name_len=self.name_lens[i]
                prefix_i=prefix[i:i+1,:,:]
                class_i=suffix[i:i+1,:name_len,:]
                suffix_i=suffix[i:i+1,name_len:,:]
                ctx_i=ctx[i:i+1,:,:]
                prompt=torch.cat([
                    prefix_i,
                    class_i,
                    ctx_i,
                    suffix_i],dim=1)
                prompts.append(prompt)
            prompts=torch.cat(prompts,dim=0)
        else:
            return ValueError
        return prompts


