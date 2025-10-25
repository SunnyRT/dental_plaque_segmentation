import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import torchvision
from typing import List, Tuple, Type, Any, Optional
# from .segment_anything.modeling.common import LayerNorm2d
# from .segment_anything.modeling import TwoWayTransformer
# from .TSGCNet import TSGCNet_decoder,TSGCNet_encoder
from segment_anything.modeling.common import LayerNorm2d
from segment_anything.modeling import TwoWayTransformer
from TSGCNet import TSGCNet_decoder,TSGCNet_encoder
from torch_geometric.nn import GATConv, global_mean_pool
import open3d as o3d
import torch_scatter


def safe_mean_pool(x, batch, batch_size):
    # x: (N, hidden_dim)
    # batch: (N,) 每个面对应哪个样本
    # batch_size: 总样本数

    # sum_pool: 对每个样本求和
    sum_pool = torch_scatter.scatter(x, batch, dim=0, dim_size=batch_size, reduce='sum')
    count = torch_scatter.scatter(torch.ones_like(batch, dtype=torch.float), batch, dim=0, dim_size=batch_size, reduce='sum')
    count = count.clamp(min=1).unsqueeze(-1)  # 防止除0
    mean_pool = sum_pool / count
    return mean_pool


#对填充的面做了处理--2d_branch_bmt4
class MeshGraphEncoder(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=256, num_heads=4):
        super().__init__()
        self.vertex_proj = nn.Linear(3, hidden_dim)
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False)
        self.face_proj = nn.Linear(hidden_dim*3, hidden_dim)
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.hidden_dim = hidden_dim


    def forward(self, mesh):
        batch_size, num_faces, _ = mesh.size()

        vertex_coords = mesh[:, :, :9].view(batch_size, num_faces, 3, 3)  # (B, F, 3, 3)

        # 1. 筛掉全零面（填充）
        non_zero_mask = (vertex_coords.abs().sum(dim=[2, 3]) > 0)  # (B, F)
        valid_vertex_coords = vertex_coords[non_zero_mask]         # (total_valid_faces, 3, 3)
        valid_count = non_zero_mask.sum(dim=1)                     # (B,)

        if valid_vertex_coords.numel() == 0:
            # 所有都是空面，返回零张量
            return torch.zeros(batch_size, self.hidden_dim, device=mesh.device)

        # 2. 投影顶点特征
        valid_vertices = valid_vertex_coords.view(-1, 3)           # (total_valid_faces * 3, 3)
        vertex_features = self.vertex_proj(valid_vertices)         # (total_valid_faces * 3, hidden_dim)

        # 3. 构造边
        total_valid_faces = valid_vertex_coords.size(0)
        face_vertices = torch.arange(3, device=mesh.device).repeat(total_valid_faces, 1)  # (N, 3)

        # 构造每个 batch 的偏移量
        vertex_counts = valid_count * 3  # 每个 batch 的有效顶点数
        vertex_offsets = torch.cat([
            torch.zeros(1, device=mesh.device, dtype=torch.long),
            torch.cumsum(vertex_counts, dim=0)[:-1]
        ])  # (B,)

        batch_indices = torch.arange(batch_size, device=mesh.device)
        face_batch = batch_indices.repeat_interleave(valid_count)         # (total_valid_faces,)
        vertex_offsets_for_faces = vertex_offsets[face_batch]            # (total_valid_faces,)

        global_vertices = vertex_offsets_for_faces.unsqueeze(1) + face_vertices  # (N, 3)

        edges = torch.stack([
            global_vertices[:, [0, 1, 1, 2, 2, 0]].flatten(),
            global_vertices[:, [1, 0, 2, 1, 0, 2]].flatten()
        ], dim=0).to(torch.long)  # (2, total_edges)

        # 4. 图注意力传播
        graph_features = F.relu(self.gat1(vertex_features, edges))
        graph_features = F.relu(self.gat2(graph_features, edges))

        # 5. 面特征聚合
        face_features = graph_features.view(total_valid_faces, 3 * self.hidden_dim)
        face_features = self.face_proj(face_features)  # (total_valid_faces, hidden_dim)

        # 6. 全局池化（每个样本池化）
        global_features = safe_mean_pool(
            face_features,             # (total_valid_faces, hidden_dim)
            face_batch,                # (total_valid_faces,)
            batch_size                 # batch 总数
        )  # (batch_size, hidden_dim)

        return global_features

#对应/home/Liting/zhanxch/SAM/checkpts/2d_branch_bmt3_fillterface
# class MeshGraphEncoder(nn.Module):
#     def __init__(self, input_dim=9, hidden_dim=256, num_heads=4):
#         super(MeshGraphEncoder, self).__init__()
        
#         # 顶点特征提取
#         self.vertex_proj = nn.Linear(3, hidden_dim)
        
#         # 图注意力层
#         self.gat1 = GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False)
#         self.gat2 = GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False)
        
#         # 面特征聚合
#         self.face_proj = nn.Linear(hidden_dim*3, hidden_dim)  # 3个顶点特征拼接
#         self.global_pool = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim)
#         )
#         self.hidden_dim = hidden_dim
        
#     def forward(self, mesh):
#         # mesh: (batch_size, 8192, 10)
#         batch_size, num_faces, _ = mesh.size()
        
#         # 提取三个顶点的坐标
#         vertex_coords = mesh[:, :, :9].view(batch_size, num_faces, 3, 3)  # (B, F, 3, 3)
#         # normals = compute_vertex_normals(vertex_coords)
        
#         # 展平为顶点列表 (batch_size * num_faces * 3, 3)
#         all_vertices = vertex_coords.reshape(-1, 3)
        
#         # 顶点特征投影
#         vertex_features = self.vertex_proj(all_vertices)  # (B*F*3, hidden_dim)
        

#         face_vertices = torch.arange(3, device=mesh.device).repeat(batch_size * num_faces, 1)  # (B*F, 3)
#         # 每个面的起始索引：i*3（i为面的序号）
#         base_idx = torch.arange(batch_size * num_faces, device=mesh.device) * 3  # (B*F,)
#         base_idx = base_idx.unsqueeze(1).repeat(1, 3)  # (B*F, 3)
#         # 全局顶点索引：base_idx + face_vertices → (B*F, 3)
#         global_vertices = base_idx + face_vertices  # 每个面的三个顶点全局索引
        
#         # 生成三角形边（0-1,1-2,2-0）及反向边（1-0,2-1,0-2）
#         edges = torch.stack([
#             global_vertices[:, [0,1,1,2,2,0]],  # 边的起点
#             global_vertices[:, [1,0,2,1,0,2]]   # 边的终点
#         ], dim=0)  # (2, B*F*6)
#         edge_index = edges.reshape(2, -1)  # 合并所有面的边 → (2, total_edges)
        
        
#         # 图卷积
#         graph_features = F.relu(self.gat1(vertex_features, edge_index))
#         graph_features = F.relu(self.gat2(graph_features, edge_index))
        
#         # 恢复面结构
#         face_features = graph_features.reshape(batch_size, num_faces, 3, -1)  # (B, F, 3, hidden_dim)
        
#         # 聚合每个面的三个顶点特征
#         face_features = face_features.reshape(batch_size, num_faces, -1)  # (B, F, hidden_dim*3)
#         face_features = self.face_proj(face_features)  # (B, F, hidden_dim)
        
#         # 全局池化获取整体几何特征
#         global_features = global_mean_pool(
#             face_features.reshape(-1, self.hidden_dim), 
#             torch.arange(batch_size, device=mesh.device).repeat_interleave(num_faces)
#         )  # (B, hidden_dim)
        
#         return global_features
    


class ImprovedBMTModule(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=256, num_prompts=1, num_heads=8):
        super(ImprovedBMTModule, self).__init__()
        
        # 几何特征提取（基于图神经网络）
        self.graph_encoder = MeshGraphEncoder(input_dim, hidden_dim)
        
        # 多尺度特征融合
        self.scale1_pool = nn.AdaptiveAvgPool1d(1)
        self.scale2_pool = nn.AdaptiveAvgPool1d(4)
        self.scale3_pool = nn.AdaptiveAvgPool1d(16)
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * (1 + 4 + 16), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 隐性提示生成
        self.learnable_tokens = nn.Parameter(torch.randn(1, num_prompts, hidden_dim))
        self.cross_attn1 = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.cross_attn2 = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
    def forward(self, mesh):

        # 提取图结构特征
        graph_features = self.graph_encoder(mesh)  # (batch_size, hidden_dim)
        # print('graph_features', graph_features.shape)
        
        # 多尺度特征提取
        batch_size, hidden_dim = graph_features.shape
        features_reshaped = graph_features.unsqueeze(2)  # (batch_size, hidden_dim, 1)
        
        scale1 = self.scale1_pool(features_reshaped).view(batch_size, -1)
        scale2 = self.scale2_pool(features_reshaped).view(batch_size, -1)
        scale3 = self.scale3_pool(features_reshaped).view(batch_size, -1)
        
        # 特征融合
        fused_features = torch.cat([scale1, scale2, scale3], dim=1)
        fused_features = self.fusion(fused_features).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # 扩展到与面数量相同的维度
        batch_size, num_faces, _ = mesh.size()
        

        expanded_features = fused_features.expand(-1, num_faces, -1)
        
        # 交叉注意力生成隐性提示
        learnable_tokens = self.learnable_tokens.expand(batch_size, -1, -1)
        
        cross_attn1_out, _ = self.cross_attn1(
            query=learnable_tokens,
            key=expanded_features,
            value=expanded_features
        )
        
        cross_attn2_out, _ = self.cross_attn2(
            query=cross_attn1_out,
            key=expanded_features,
            value=expanded_features
        )
        
        return cross_attn2_out

class SASOTA(nn.Module):
    def __init__(self, sam):
        super().__init__()
        self.resize = torchvision.transforms.Resize(
        (1024, 1024),
        interpolation=torchvision.transforms.InterpolationMode.NEAREST
        )
        self.image_encoder = sam.image_encoder
        self.preprocess = sam.preprocess
        self.prompt_encoder = sam.prompt_encoder 
        
        hidden_dim = 256
        self.bmt = ImprovedBMTModule(hidden_dim=hidden_dim, num_prompts = 1)
        self.adapter = nn.Linear(hidden_dim, sam.mask_decoder.transformer_dim)

        # self.encoder_3d = TSGCNet_encoder(in_channels=9, output_channels=2, k=12)
        #  self.encoder_3d = TSGCNet_encoder(in_channels=9, output_channels=2)
        
        # self.fusion = TwoWayTransformer(
        #     dim=256,
        #     depth=6,
        #     num_heads=8,
        #     mlp_ratio=4.0,
        #     qkv_bias=True,
        #     norm_layer=LayerNorm2d
        # )
        self.projector_3d = nn.Conv2d(2, 256, kernel_size=1, stride=1, padding=0) # TODO: project TSGCNet encoder output to dense prompt embedding dimension
        # self.decoder_3d = TSGCNet_decoder(in_channels=9, output_channels=2, k=k)
        self.mask_decoder = sam.mask_decoder
        # prompt_embed_dim = 256 #7自己加的
        # self.mask_decoder =MaskDecoder(
        #     # num_multimask_outputs=3,
        #     num_multimask_outputs=1,
        #     transformer=TwoWayTransformer(
        #         depth=2,
        #         embedding_dim=prompt_embed_dim,
        #         mlp_dim=2048,
        #         num_heads=8,
        #     ),
        #     transformer_dim=prompt_embed_dim,
        #     iou_head_depth=3,
        #     iou_head_hidden_dim=256,
        # )

        self._freeze_encoder()

        self.mask_decoder.load_state_dict(sam.mask_decoder.state_dict())
        self.prompt_encoder.load_state_dict(sam.prompt_encoder.state_dict())

        
    # def forward(self, image):#,mesh):
    def forward(self, image, mesh):#,mesh):
        
        image = image.to(next(self.parameters()).dtype)
        mesh = mesh.to(next(self.parameters()).dtype)

        image = self.resize(image)
        input_images = self.preprocess(image)
        # print('input_images.shape',input_images.shape) #[b, 3, 1024, 1024]
        image_embeddings = self.image_encoder(input_images)

        sparse_embeddings,dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=None,
            masks=None
            )
        
        # 生成隐性提示
        implicit_prompts = self.bmt(mesh)
        # print('implicit_prompts', implicit_prompts.shape) #[4, 5, 256]
        # 适配到SAM的维度
        adapted_prompts = self.adapter(implicit_prompts)
        # 使用SAM的mask decoder生成掩码
        sparse_embeddings = adapted_prompts.permute(1, 0, 2) 
        # print('sparse_embeddings', sparse_embeddings.shape) ##[5, 4, 256]
        
        

        # dense_embeddings = self.encoder_3d(mesh)
        # dense_embeddings = self.projector_3d(dense_embeddings)
        # print('image_embeddings', image_embeddings.shape) #[b, 256, 64, 64]
        # print('self.prompt_encoder.get_dense_pe()', self.prompt_encoder.get_dense_pe().shape) #[b, 256, 16, 16]
        # print('dense_prompt_embeddings', dense_embeddings.shape) #[b, 256, 16, 16]
        # print('sparse_prompt_embeddings', sparse_embeddings.shape) #[b, 0, 256]
        low_res_masks, iou_predictions = self.mask_decoder( 
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            dense_prompt_embeddings=dense_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            multimask_output = False
        )
        # print('low_res_masks', low_res_masks.shape) #[b, 1, 256, 256]
        # low_res_masks = low_res_masks[:, 0:1, :, :]
        # low_res_masks = F.interpolate(
        #     low_res_masks,
        #     size=(256, 256),
        #     mode="bilinear",
        #     align_corners=False
        # )  # [B, 3, 256, 256]
        
        # print('a', self.mask_decoder)
        # TODO: maybe back project the 2d Mask to 3d space so that you can constraint the fused 3d mask with the ground truth
        
        # pred_3d = self.decoder_3d(fuse_features)
        # print('low_res_masks', low_res_masks.shape) #torch.Size([4, 3, 64, 64])

        outputs = {
            'pred_masks': low_res_masks,
            'pred_3d': None,
            'iou_predictions': iou_predictions,
            'low_res_logits': None,
            
        }
        return outputs
        # return 
    
    def _freeze_encoder(self):
        """
        Freeze the encoder parameters.
        """
        last_layer_no = 176
        for layer_no, param in enumerate(self.image_encoder.parameters()):
            if(layer_no > (last_layer_no - 6)):
                param.requires_grad = True
            else:
                param.requires_grad = False



class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        tranformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        # if multimask_output:
        #     mask_slice = slice(1, None)
        # else:
        #     mask_slice = slice(0, 1)
        # masks = masks[:, mask_slice, :, :]
        # iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)  # [b, c, token_num]

        b, c, h, w = upscaled_embedding.shape  # [h, token_num, h, w]
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)  # [1, 4, 256, 256], 256 = 4 * 64, the size of image embeddings

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred,src


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
