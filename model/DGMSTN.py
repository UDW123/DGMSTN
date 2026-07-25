import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None
try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None




class SpatioConvLayer(nn.Module):
    def __init__(self, ks, c_in, c_out, device):
        super(SpatioConvLayer, self).__init__()
        self.ks = ks
        self.device = device

        # 图卷积核参数 [c_in, c_out, ks*2]
        self.theta = nn.Parameter(torch.FloatTensor(c_in, c_out, ks * 2).to(device))
        self.b = nn.Parameter(torch.FloatTensor(1, 1, c_out).to(device))
        # self.theta = nn.Parameter(torch.FloatTensor(c_in, c_out, ks * 2).to(device))
        # self.b = nn.Parameter(torch.FloatTensor(1, c_out, 1).to(device))  # no time dim!
        self.align = Align(c_in, c_out)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x, G_dict):
        """
        x: [B, N, C_in]
        G_dict: {'dataset': [A1, A2, ...]}, 每个 A: [N, N]
        """
        x = x.to(self.device)
        B, N, C_in = x.shape

        # Step 1. 构建切比雪夫多项式基
        support_set = []
        for support in G_dict['dataset']:
            support_ks = [torch.eye(support.shape[0]).to(self.device), support]
            for k in range(2, self.ks):
                support_ks.append(torch.matmul(2 * support, support_ks[-1]) - support_ks[-2])
            support_set.append(torch.stack(support_ks))  # [ks, N, N]
        support_set = torch.stack(support_set)           # [num_support, ks, N, N]

        # Step 2. 图卷积传播
        # 遍历每个支持矩阵
        x_c = []
        for support in support_set:
            # support: [ks, N, N]
            # x: [B, N, C_in]
            x_tmp = torch.einsum("knm,bmc->bknc", support, x)  # [B, ks, N, C_in]
            x_c.append(x_tmp)
        x_c = torch.cat(x_c, dim=1)  # [B, ks*num_support, N, C_in]

        # Step 3. 特征融合：C_in -> C_out
        # θ: [C_in, C_out, ks*2]  (为了兼容 ks*num_support)
        K = x_c.shape[1]
        theta = self.theta[:, :, :K]  # 截取匹配长度
        x_gc = torch.einsum("iok,bkni->bno", theta, x_c)  # [B, N, C_out, K]
        x_gc = x_gc + self.b  # 在所有 Ks 上求和 -> [B, N, C_out]

        # Step 4. 残差连接
        x_in = self.align(x)  # [B, N, C_out]
        return torch.relu(x_gc + x_in)



class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:  # Doesn't support outputting the states
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class Align(nn.Module):
    """用于匹配输入输出通道数"""
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        if c_in != c_out:
            self.proj = nn.Linear(c_in, c_out)
        else:
            self.proj = None

    def forward(self, x):
        # x: [B, N, C_in]
        if self.proj is not None:
            x = self.proj(x)
        return x


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):

        x = self.value_embedding(x)
        return self.dropout(x)

class MambaBlock(nn.Module):
    def __init__(self, mamba, mamba_r,cin, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(MambaBlock, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.DataEmbedding = DataEmbedding_inverted(cin, d_model, dropout=dropout)
        self.mamba = mamba
        self.mamba_r = mamba_r
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
    def forward(self, x):
        # x = x.permute(0, 2, 1)
        if(self.mamba_r==None):
            new_x = self.mamba(x)
        elif(self.mamba==None):
            new_x = self.mamba_r(x.flip(dims=[1])).flip(dims=[1])
        else:
            new_x = self.mamba(x) + self.mamba_r(x.flip(dims=[1])).flip(dims=[1])
        attn =1

        x = x + new_x
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn



class MMTest(nn.Module):

    def __init__(self, DEVICE, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, len_input,adj_mx,num_of_vertices,mem_num=20,mem_dim=64):
        super(MMTest, self).__init__()

        self.mem_num = mem_num;
        self.mem_dim = mem_dim;
        self.num_nodes = num_of_vertices
        self.memory = self.construct_memory()

        self.enc_embedding = DataEmbedding_inverted(len_input, 256,
                                                    0.1)
        self.MADGC = SpatioConvLayer(K,c_in=256,c_out=256,device='cuda')
        self.adj_mx=adj_mx
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))

        self.ln = nn.LayerNorm(nb_time_filter)
        self.mambablock1 = MambaBlock(
                        Mamba(
                            d_model=256,  # Model dimension d_model
                            d_state=32,  # SSM state expansion factor
                            d_conv=2,  # Local convolution width
                            expand=1,  # Block expansion factor)
                        ),
            Mamba(
                d_model=256,  # Model dimension d_model
                d_state=32,  # SSM state expansion factor
                d_conv=2,  # Local convolution width
                expand=1,  # Block expansion factor)
            ),
                    256,
                    256,
                    dropout=0.1,
                    activation='gelu'
                )

        self.mambablock2 = MambaBlock(
            Mamba(
                d_model=256,  # Model dimension d_model
                d_state=32,  # SSM state expansion factor
                d_conv=2,  # Local convolution width
                expand=1,  # Block expansion factor)
            ),
            Mamba(
                d_model=256,  # Model dimension d_model
                d_state=32,  # SSM state expansion factor
                d_conv=2,  # Local convolution width
                expand=1,  # Block expansion factor)
            ),
            256,
            256,
            dropout=0.1,
            activation='gelu'
        )
        self.projector = nn.Linear(256, len_input, bias=True)
        self.projector1 = nn.Linear(nb_time_filter, in_channels, bias=True)
        self.projector2 = nn.Linear(in_channels, nb_time_filter, bias=True)
        self.device=DEVICE

    def construct_memory(self):
        memory_dict = nn.ParameterDict()
        # 通过注意力机制，用来计算特征相似性
        # 其中Memory用来存储学习到的历史特征，方便后模型学习，模型可以通过查询这些记忆项来辅助当前的预测或决策。
        memory_dict['Memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)  # (M, d)
        # memory_dict['Wq'] = nn.Parameter(torch.randn(self.rnn_units, self.mem_dim),
        #                                  requires_grad=True)  # project to query
        # 节点嵌入后续为了计算节点相似性，
        memory_dict['We1'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num),
                                          requires_grad=True)  # project memory to embedding
        memory_dict['We2'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num),
                                          requires_grad=True)  # project memory to embedding
        """
        遍历 memory_dict 中的所有参数，并使用 Xavier 初始化方法对每个参数进行初始化。
        Xavier 初始化有助于加速训练过程，并有助于保持激活函数输入的方差在整个网络中一致。
        """
        for param in memory_dict.values():
            nn.init.xavier_normal_(param)
        return memory_dict

    def forward(self, x):

        node_embeddings1 = torch.matmul(self.memory['We1'], self.memory['Memory'])
        node_embeddings2 = torch.matmul(self.memory['We2'], self.memory['Memory'])
        g1 = F.softmax(F.relu(torch.mm(node_embeddings1, node_embeddings2.T)), dim=-1)
        g2 = F.softmax(F.relu(torch.mm(node_embeddings2, node_embeddings1.T)), dim=-1)
        supports = [g1, g2]
        G_dict = {}
        G_dict['dataset'] = supports;

        x0 = x
        x0 = x0.permute(0, 2, 1)
        # if x0.dim() < 4:
        #     x0 = x0.permute(0, 2, 1)
        #     x0 = torch.unsqueeze(x0, dim=2)

        batch_size, num_of_vertices, num_of_timesteps = x0.shape
        x1 = torch.squeeze(x0, dim=2).to(self.device)
        enc_out = self.enc_embedding(x1)  # DataEmbedding_inverted(len_input, 512,0.1)(B,N,512)
        mamba_output, ATT = self.mambablock1(enc_out)
        output_gcn = self.MADGC(mamba_output,G_dict)#(B,N,inchannel) STGCNLayer(len_input, nb_chev_filter, K)
        mamba_output,ATT = self.mambablock2(output_gcn)# encoder(encoderLayer(mamba(512,32,2,1),mamba,512,2048,0.1,gelu),LayerNorm(512))

        mamba_output = self.projector(mamba_output).permute(0, 2, 1)[:, :, :num_of_vertices]

        mamba_output=mamba_output.permute(0, 2, 1)

        mamba_output=torch.unsqueeze(mamba_output, dim=3)

        mamba_output=self.projector2(mamba_output)

        # residual shortcut
        x0 = torch.unsqueeze(x0, dim=2)
        x_residual = self.residual_conv(x0.permute(0, 2, 1, 3))

        x_residual=x_residual.permute(0, 2, 3, 1)

        x_residual1 = self.ln(F.relu(x_residual + mamba_output))

        x_residual2=self.projector1(x_residual1)

        x_residual2=x_residual2.permute(0,1,3,2)

        return x_residual2


class DGMSTN(nn.Module):

    def __init__(self, DEVICE, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, num_for_predict, len_input,adj_mx,num_of_vertices):

        super(DGMSTN, self).__init__()

        self.Block = MMTest(DEVICE, in_channels, K, nb_chev_filter, nb_time_filter, time_strides ,len_input,adj_mx,num_of_vertices)
        self.DEVICE = DEVICE
        self.projector3 = nn.Linear(len_input, num_for_predict, bias=True)
        self.to(DEVICE)

    def forward(self, x):

        x = self.Block(x)
        output=torch.squeeze(x, dim=2)
        output= self.projector3(output)
        output_final=output.permute(0,2,1)
        return output_final



def make_model(DEVICE,  in_channels, K, nb_chev_filter, nb_time_filter, time_strides, adj_mx, num_for_predict, len_input,num_of_vertices,args):

    model = DGMSTN(DEVICE,  in_channels, K, nb_chev_filter, nb_time_filter, time_strides , num_for_predict, len_input,adj_mx,num_of_vertices)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model