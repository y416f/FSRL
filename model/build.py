from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import random

class FSRL(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 
        
        self.discriminator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
            )

        if 'id' in args.loss_names:
            ''' 
            self.classifier = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, self.num_classes))]))
            nn.init.normal_(self.classifier.dense.weight, std=0.001)
            nn.init.constant_(self.classifier.dense.bias, val=0.0)
            nn.init.normal_(self.classifier.fc.weight, std=0.001)
            nn.init.constant_(self.classifier.fc.bias, val=0.0) 
            '''           
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5
            
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

            ##############pgu##############
            self.query_embed_image = nn.Parameter(torch.randn(1, 6, 512))      #pgu
            self.cross_attn_pgu = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            nn.init.normal_(self.cross_attn_pgu.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn_pgu.out_proj.weight, std=proj_std)
            self.ln_t_pgu = LayerNorm(self.embed_dim)
            self.ln_i_pgu = LayerNorm(self.embed_dim)
            self.ln_p_pgu = LayerNorm(self.embed_dim)

            self.fc_layers = nn.ModuleList([nn.Linear(512, 512) for _ in range(6)])

            self.mlm_head_6 = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, 6))]))

            ##############pgu##############

            #########error modeling###########
            self.cross_attn_error = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.ln_pre_t_e = LayerNorm(self.embed_dim)
            self.ln_pre_i_e = LayerNorm(self.embed_dim)
            #init cross attn
            nn.init.normal_(self.cross_attn_error.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn_error.out_proj.weight, std=proj_std)
            '''
            self.mlm_head_e = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, 2))]))
            '''
            self.mlm_head_e = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
                )
            # init mlm head
            #nn.init.normal_(self.mlm_head_e.dense.weight, std=fc_std)
            #nn.init.normal_(self.mlm_head_e.fc.weight, std=proj_std)
            ##########error modeling##########

            ##########global cat###########
            #self.cross_gcat_linear = nn.Linear(512, 512)
            ##########global cat###########
            
    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    
    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def cross_former_error(self, q, k, v):
        x = self.cross_attn_error(
                self.ln_pre_t_e(q),
                self.ln_pre_i_e(k),
                self.ln_pre_i_e(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        return x[:, 0, :].float()
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()


    def pgu_encode_i(self,i_feature):
        query_embed = self.query_embed_image.repeat(i_feature.size(0), 1, 1).half()
        i_pgu = self.cross_attn_pgu(self.ln_p_pgu(query_embed),
                self.ln_i_pgu(i_feature),
                self.ln_i_pgu(i_feature),
                need_weights=False)[0]
        output_i = []
        for i, fc_layer in enumerate(self.fc_layers):
            i_pgu_t= i_pgu[:, i, :]             
            output_i.append(fc_layer(i_pgu_t))  
        #reshaped_i = torch.reshape(torch.stack(output_i, dim=1), (self.batch_size, -1))
        #output_i_end = self.ln_end_pgu(reshaped_i).float()

        #output_i_end_mix = self.mix_layers(torch.cat((fi, output_i_end), dim=1).half()).float()

        return output_i


    def pgu_encode_t(self,t_feature):
        query_embed = self.query_embed_image.repeat(t_feature.size(0), 1, 1).half()
        t_pgu = self.cross_attn_pgu(self.ln_p_pgu(query_embed),
                self.ln_t_pgu(t_feature),
                self.ln_t_pgu(t_feature),
                need_weights=False)[0]
        output_t = []
        for i, fc_layer in enumerate(self.fc_layers):
            t_pgu_t= t_pgu[:, i, :]             
            output_t.append(fc_layer(t_pgu_t))  
        #reshaped_t = torch.reshape(torch.stack(output_t, dim=1), (self.batch_size, -1))
        #output_t_end = self.ln_end_pgu(reshaped_t).float()

        #output_t_end_mix = self.mix_layers(torch.cat((ft, output_t_end), dim=1).half()).float()

        return output_t


    def forward(self, batch,epoch,store1,store2,idstore):
        ret = dict()
        images = batch['images']
        caption_ids = batch['caption_ids'] ###############原始##############
        #caption_ids = batch['mlm_ids']
        #########error modeling#########
        t_index = caption_ids.argmax(dim=-1)

        image_feats, text_feats = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
        
        gi_feats = F.normalize(i_feats, dim=1)
        gt_feats = F.normalize(t_feats, dim=1)

        i_dis = self.discriminator((gi_feats).half()).float()
        t_dis = self.discriminator((gt_feats).half()).float()
        y_real = torch.ones(i_feats.size(0)).to(i_feats.device)
        y_fake = torch.zeros(i_feats.size(0)).to(i_feats.device)
        BCE_loss = nn.BCEWithLogitsLoss()
        D_real_loss = BCE_loss(i_dis.squeeze(), y_real)
        D_fake_loss = BCE_loss(t_dis.squeeze(), y_fake)
        D_train_loss = D_real_loss + D_fake_loss
        ret.update({'adv_loss':D_train_loss})
        
        BCE_loss = nn.BCEWithLogitsLoss()
        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        if 'itc' in self.current_task:
            ret.update({'itc_loss':objectives.compute_itc(i_feats, t_feats, logit_scale)})
        
        if 'sdm' in self.current_task:
            ret.update({'sdm_loss':objectives.compute_js_sdm(i_feats, t_feats, batch['pids'], logit_scale)})
            
            ##############pgu##############
             
            output_i = self.pgu_encode_i(image_feats.detach())
            output_t = self.pgu_encode_t(text_feats.detach())
            output_i_c = torch.cat(output_i,dim=1).float()
            output_t_c = torch.cat(output_t,dim=1).float()
            ret.update({'xld_loss':objectives.compute_js_sdm(output_i_c, output_t_c, batch['pids'], logit_scale)*self.args.id_loss_weight})
            
            total_6 = 0
            for i in range(6):
                image_6 = self.mlm_head_6(output_i[i]).float()
                text_6 = self.mlm_head_6(output_t[i]).float()
                lable_6 = (torch.ones(i_feats.size(0),dtype=torch.int) * i).long().to(i_feats.device)
                total_6 += (objectives.compute_mlm_6(image_6,lable_6) + objectives.compute_mlm_6(text_6,lable_6))
            ret.update({'hc_loss':total_6})
            
        
        if 'id' in self.current_task:
            numm = i_feats.size(0)
            zero_tensor_i = torch.zeros(numm, 11003).to(i_feats.device)
            zero_tensor_t = torch.zeros(numm, 11003).to(i_feats.device)
            for i, index in enumerate(batch['pids']):
                        zero_tensor_i[i, index] = 0.4
                        zero_tensor_t[i, index] = 0.4

            image_logits = self.classifier(i_feats.half()).float()
            text_logits = self.classifier(t_feats.half()).float()

            if epoch == 1:
                ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])*self.args.id_loss_weight})
            
            else:
                store_i = torch.cat(store1,dim=0)   ########(68000,6,512)
                store_t = torch.cat(store2,dim=0)
                store_id = torch.cat(idstore,dim=0)
                for i in range(6):
                    sim1 = F.normalize(output_i[i], p=2, dim=1) @ F.normalize(store_i[:,i,:], p=2, dim=1).t()  ######(64,68000)
                    sim2 = F.normalize(output_t[i], p=2, dim=1) @ F.normalize(store_t[:,i,:], p=2, dim=1).t()

                    topk_values_i, topk_indice_i = torch.topk(sim1, k=100, dim=1)  #######(64,75)
                    topk_values_t, topk_indice_t = torch.topk(sim2, k=100, dim=1)

                    shot_i = torch.gather(store_id.unsqueeze(0).repeat(numm, 1), dim=1, index=topk_indice_i) ######(64,75)
                    shot_t = torch.gather(store_id.unsqueeze(0).repeat(numm, 1), dim=1, index=topk_indice_t)

                    increment_tensor = (torch.ones_like(zero_tensor_i) * 0.001).to(i_feats.device)
                    soft_lable_i = zero_tensor_i + torch.scatter_add(torch.zeros_like(zero_tensor_i).to(i_feats.device), dim=1, index=shot_i, src=increment_tensor)
                    #soft_lable_i = zero_tensor_i + torch.scatter_add(torch.zeros_like(zero_tensor_i).to(i_feats.device), dim=1, index=shot_i, src=increment_tensor) + torch.scatter_add(torch.zeros_like(zero_tensor_i).to(i_feats.device),dim=1,index=shot_t, src=increment_tensor)
                    #soft_lable_t = soft_lable_i
                    soft_lable_t = zero_tensor_t + torch.scatter_add(torch.zeros_like(zero_tensor_t).to(i_feats.device), dim=1, index=shot_t, src=increment_tensor)
                
                hard = objectives.compute_id(image_logits, text_logits, batch['pids'])*self.args.id_loss_weight * 0.5
                ret.update({'id_loss':hard + objectives.compute_soft_id(image_logits, text_logits, soft_lable_i, soft_lable_t)*self.args.id_loss_weight * 0.5})
            

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})
        
        if 'mlm' in self.current_task:
            mlm_ids = batch['mlm_ids']      ########原始########
            mlm_feats = self.base_model.encode_text(mlm_ids)
            #o_text = self.base_model.encode_text(batch['caption_ids'])   #####add
            #mlm_feats = text_feats                                       #####add
            x = self.cross_former(mlm_feats, image_feats, image_feats)
            
            mlme_pre = (self.mlm_head_e(x)).reshape(-1)#################

            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]
            scores = x.float().reshape(-1, self.args.vocab_size)

            mlm_labels = batch['mlm_labels'].reshape(-1)

            error_labels0 = torch.where(mlm_labels != 0, 1, mlm_labels)##############
            loss_pre0 = BCE_loss(mlme_pre.float(), error_labels0.float())############ 
            ret.update({'mlm2_loss': loss_pre0*self.args.mlm_loss_weight})############

            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})

            scores = x.float().reshape(-1, self.args.vocab_size)

            pred = scores.max(1)[1] #一维tesor 616
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})
        
            ######################error modeling##################
            error_list = []
            for i in range(batch['mlm_ids'].size(0)):
                mlm_i = batch['mlm_ids'][i].clone()
                mlm_lain = torch.nonzero(batch['mlm_labels'][i]).squeeze()
                top_values, top_indices = torch.topk(x[i].float(), 32, dim=1, largest=True, sorted=True)
                random_indices = torch.randint(low=0, high=32, size=(77,))
                # 对 tensor 进行切片
                result = top_indices[torch.arange(77), random_indices]
                mlm_i[mlm_lain] = result[mlm_lain]
                if mlm_lain.dim() == 0:
                    mlm_lain = mlm_lain.unsqueeze(0)
                for index in mlm_lain:
                    if mlm_i[index].item() == batch['caption_ids'][i][index].item():
                        if top_indices[index][0].item() == batch['caption_ids'][i][index].item():
                            mlm_i[index] = top_indices[index][1]
                        else:
                            mlm_i[index] = top_indices[index][0]
                error_list.append(mlm_i)

            error_text = torch.stack(error_list)

            error_feats = self.base_model.encode_text(error_text)
            error_x = self.cross_former_error(error_feats, image_feats, image_feats)
            '''
            error_pre = (self.mlm_head_e(error_x)).reshape(-1, 2)
            
            error_x = self.mlm_head(error_x)
            scores_e = error_x.float().reshape(-1, self.args.vocab_size)

            error_labels1 = torch.where(mlm_labels != 0, 1, mlm_labels)
            error_labels2 = error_labels1 ^ 1
            error_labels = torch.cat((error_labels1.unsqueeze(1), error_labels2.unsqueeze(1)), dim=1)

            loss_pre = F.cross_entropy(error_pre, error_labels.float())/0.1
            '''
            error_pre = (self.mlm_head_e(error_x)).reshape(-1)
            
            error_x = self.mlm_head(error_x)
            scores_e = error_x.float().reshape(-1, self.args.vocab_size)

            error_labels1 = torch.where(mlm_labels != 0, 1, mlm_labels)
            loss_pre = BCE_loss(error_pre.float(), error_labels1.float())

            ret.update({'mlmpre_loss': loss_pre*self.args.mlm_loss_weight})
            ret.update({'mlmerror_loss': objectives.compute_mlm(scores_e, mlm_labels)*self.args.mlm_loss_weight})
            ###########################
            '''
            ###########global cat############
            add_feats = mlm_feats + image_feats[:, 0, :].unsqueeze(1)
            add_feats = self.cross_gcat_linear(add_feats)
            add_feats = add_feats.permute(1, 0, 2)
            add_feats = self.cross_modal_transformer(add_feats)
            add_feats = add_feats.permute(1, 0, 2)
            add_feats = self.ln_post(add_feats)
            add_feats = self.mlm_head(add_feats)
            scores = add_feats.float().reshape(-1, self.args.vocab_size)
            #mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({'mlm2_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})
            ##################################
            '''
            '''
            remain_image_feats = image_feats[:, 1:, :]
            #query_embed = self.query_embed_image.repeat(image_feats.size(0), 1, 1).half()

            output_i_end = self.pgu_encode_i(remain_image_feats,i_feats)
            output_t_end = self.pgu_encode_t(mlm_feats,t_feats)

            #output_i_end = self.ln_end_pgu(reshaped_i).float()  
            #output_t_end = self.ln_end_pgu(reshaped_t).float()
            ret.update({'xld_loss':objectives.compute_sdm(output_i_end, output_t_end, batch['pids'], logit_scale)})
            '''
            ##############pgu##############
        

        return ret, torch.stack(output_i, dim=1), torch.stack(output_t, dim=1)


def build_model(args, num_classes=11003):
    model = FSRL(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
