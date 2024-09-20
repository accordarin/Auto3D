import torch
import torch.nn as nn
from models.models_3d.schnet_slot import slotSchNet
from torch_geometric.nn import global_mean_pool



def print_issue(name, tensor):
    if torch.any(torch.isnan(tensor)) or torch.any(torch.isinf(tensor)):
        print(f"{name}: {tensor.shape}")
        print(f" -> min: {tensor.min()}, max: {tensor.max()}, has nan: {torch.any(torch.isnan(tensor))}, "
              f"has inf: {torch.any(torch.isinf(tensor))}")


class SlotGATLayer(nn.Module):
    """
    Implementation of the slotGAT message passing layer.

    __init__ Args:
        edge_feat: Dimension of edge features.
        num_edge_types: Number of edge types.
        in_feat: Dimension of input features.
        out_feat: Dimension of output features.
        num_heads: Number of attention heads.
        feat_drop: Feature dropout rate.
        attn_drop: Attention dropout rate.
        negative_slope: Negative slope of LeakyReLU.
        residual: Whether to use residual connection.
        activation: Activation function to process output (optional).
        bias: Bias added to output (optional).
        alpha: Residual attention parameter (optional).
        num_node_types: Number of node types.
        input_head: Whether to use input head.
    """
    def __init__(self,
                 edge_feat,
                 num_edge_types,
                 in_feat,
                 out_feat,
                 num_heads,
                 feat_drop: float = 0.,
                 attn_drop: float = 0.,
                 negative_slope = 0.2,
                 residual: bool = False,
                 activation = None,
                 bias: bool = False,
                 alpha: float = 0.,
                 num_node_types: int = 0,
                 input_head = False,
                 device=f'cuda:{0}'
                 ):
        super().__init__()
        self.edge_feat = edge_feat
        self.num_heads = num_heads

        # Size of source and destination features
        self.in_feat = in_feat
        self.out_feat = out_feat

        # Embedding for edge features
        self.edge_embed = nn.Embedding(num_edge_types, edge_feat) if edge_feat else None

        self.num_node_types = num_node_types
        self.attentions = None

        # Fully connected layers
        self.fc = nn.Parameter(torch.FloatTensor(size=(num_node_types, in_feat, out_feat * num_heads)),
                               requires_grad=True)

        # For edge features
        self.fc_edge_feat = nn.Linear(in_features=edge_feat,
                                      out_features=edge_feat * num_heads,
                                      bias=False) if edge_feat else None

        # Attention mechanisms for source and destination nodes
        # One vector for each head, with size out_feat for each slot
        self.attn_src = nn.Parameter(
            torch.FloatTensor(size=(1, num_heads, out_feat * num_node_types)), requires_grad=True)
        self.attn_dst = nn.Parameter(
            torch.FloatTensor(size=(1, num_heads, out_feat * num_node_types)), requires_grad=True)

        # For edge features
        self.attn_edge = nn.Parameter(
            torch.FloatTensor(size=(1, num_heads, edge_feat)), requires_grad=True) if edge_feat else None

        # Regularization layers (prevents overfitting)
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        # Nonlinear activation function
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        # Setup residual connection (allows the original input to be added back to the output for training stability)
        if residual:
            if self.in_feat != self.out_feat:
                self.residual_fc = nn.Linear(in_feat, num_heads * out_feat, bias=False)
            else:
                self.residual_fc = nn.Identity()
        else:
            self.register_buffer('residual_fc', None)

        self.reset_parameters()

        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(torch.zeros((1, num_heads, out_feat)), requires_grad=True)
        self.alpha = alpha
        self.input_head = input_head
        self.device = device
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')

        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc, gain=gain)
        else:
            raise NotImplementedError()

        nn.init.xavier_normal_(self.attn_src, gain=gain)
        nn.init.xavier_normal_(self.attn_dst, gain=gain)

        if self.edge_feat:
            nn.init.xavier_normal_(self.attn_edge, gain=gain)

        if isinstance(self.residual_fc, nn.Linear):
            nn.init.xavier_normal_(self.residual_fc.weight, gain=gain)
        elif isinstance(self.residual_fc, nn.Identity):
            pass
        elif isinstance(self.residual_fc, nn.Parameter):
            nn.init.xavier_normal_(self.residual_fc, gain=gain)

        if self.edge_feat:
            nn.init.xavier_normal_(self.fc_edges.weight, gain=gain)

    def forward(self,
                edge_idx,
                node_feats,
                edge_attr=0,
                res_attn=None):
        """
        Message-passing layer from SlotGAT paper.

        Args:
            edge_idx: Edge indices of shape (2, num_edges_per_conformer * num_confs).
            node_feats: Node feature vectors of shape (num_confs, num_nodes, in_feat).
            edge_attr: Edge attributes, UNUSED.
            res_attn: Residual attention weights to combine with source, dest, edge attention weights (optional).

        Returns:
            outs: Final node features of shape (num_confs, num_nodes, num_node_types, out_feat).
            a: Attention weights of shape (num_confs, num_nodes, num_nodes, num_heads).
        """

        # Get info for reshaping
        num_confs, num_nodes, _, _ = node_feats.shape

        # Apply feature dropout for regularization
        h_src = self.feat_drop(node_feats)  # (num_confs, num_nodes, in_feat)

        # Process edge features (optional)
        if edge_attr != 0:
            e_feat = self.edge_emb(edge_attr)  # Embed edge features
            e_feat = self.fc_e(e_feat).view(-1, self._num_heads, self._edge_feats)
            a_edge = (e_feat * self.attn_e).sum(dim=-1).unsqueeze(-1)  # Edge attention scores
        else:
            a_edge = 0

        # Apply linear transformation to input
        h_src = h_src.view(-1, self.num_node_types, self.in_feat)  # (num_confs * num_nodes, num_node_types, in_feat)
        feat_dst = torch.einsum('bji,jik->bjk', h_src, self.fc)
        feat_dst = feat_dst.view(num_confs, num_nodes, self.num_node_types, self.out_feat * self.num_heads)
        # ^ (num_confs, num_nodes, num_node_types, out_feat * num_heads)
        feat_dst = feat_dst.permute(0, 2, 1, 3).reshape(
            num_confs, -1, self.num_node_types, self.num_heads, self.out_feat
        ).permute(0, 1, 3, 2, 4).flatten(3)  # (num_confs, num_nodes, num_heads, num_node_types * out_feat)
        feat_src = feat_dst = feat_dst.reshape(-1, self.num_heads, self.out_feat * self.num_node_types)

        # Mask processed features by source and destination
        mask_src = torch.zeros(num_confs * num_nodes, self.num_heads, self.num_node_types * self.out_feat).to(self.device)
        mask_dst = torch.zeros_like(mask_src).to(self.device)
        mask_src[edge_idx[0]] = 1
        mask_dst[edge_idx[1]] = 1
        feat_src = feat_src * mask_src  # (num_confs * num_nodes, num_heads, num_node_types * out_feat)
        feat_dst = feat_dst * mask_dst

        # Compute source, destination attention weights
        print_issue("attn_src", self.attn_src)
        print_issue("attn_dst", self.attn_dst)
        a_src = (feat_src * self.attn_src).view(num_confs, num_nodes, self.num_heads, self.out_feat * self.num_node_types)
        a_src = torch.clamp(a_src, min=-1e9, max=1e9)
        print_issue("a_src", a_src)
        a_dst = (feat_dst * self.attn_dst).view(num_confs, num_nodes, self.num_heads, self.out_feat * self.num_node_types)
        a_dst = torch.clamp(a_dst, min=-1e9, max=1e9)
        print_issue("a_dst", a_dst)
        a_src, a_dst = a_src.sum(dim=-1), a_dst.sum(dim=-1)
        print_issue("a_src after summing", a_src)
        print_issue("a_dst after summing", a_src)
        a_src, a_dst = a_src.unsqueeze(2), a_dst.unsqueeze(1)
        # a_src: (num_confs, num_nodes, 1, num_heads)
        # a_dst: (num_confs, 1, num_nodes, num_heads)
        print_issue("a_src after unsqueeze", a_src)
        print_issue("a_dst after unsqueeze", a_dst)

        # Get final attention weights
        a = a_src + a_dst
        print_issue("a", a)
        a = a + a_edge
        print_issue("a after adding a_edge", a)
        a = self.leaky_relu(a)
        print_issue("a after leaky ReLU", a)
        a = torch.clamp(a, min=-1e9, max=1e9)
        a = nn.functional.softmax(a, dim=0)
        print_issue("a after softmax", a)
        a = self.attn_drop(a)  # (num_confs, num_nodes, num_nodes, num_heads)
        print_issue("a after attn_drop", a)

        # Combine current attention weights with residual attention
        if res_attn is not None:
            a = a * (1 - self.alpha) + res_attn * self.alpha

        # Compute messages, update node features
        feat_dst = feat_dst.view(num_confs, num_nodes, self.num_heads, self.num_node_types * self.out_feat)
        m = torch.einsum('cvuh,cuhf->cvhf', a, feat_dst)  # (num_confs, num_nodes, num_heads, num_node_types * out_feat)
        print_issue("m", m)
        m = nn.functional.relu(m)
        print_issue("m after ReLU", m)
        m = m.view(num_confs, num_nodes, self.num_heads, self.num_node_types, self.out_feat)
        outs = m.sum(dim=2)  # (num_confs, num_nodes, num_node_types, out_feat)
        print_issue("outs", outs)

        # Residual connection
        # TODO: fix this
        if self.residual_fc is not None:
            if self.in_dest_feats != self.out_feat:  # If input and output feature dimensions differ
                res_value = torch.bmm(h_src, self.residual_fc)

                # shape = (num_nodes*num_heads*(num_ntype*hidden_dim))
                res_value = res_value.permute(0, 2, 1, 3).view(
                    num_confs, -1, self.num_node_types, self.num_heads, self.out_feat
                ).permute(0, 1, 3, 2, 4).flatten(2)
            else:
                # Identity
                res_value = self.residual_fc(h_src).view(
                    h_src.shape[1], -1, self.out_feat * self.num_node_types
                )
            outs = outs + res_value

        if self.bias:
            outs = outs + self.bias

        if self.activation:
            outs = self.activation(outs)

        # Return final node features, attention coefficients
        return outs, a

class SlotGATAttention(nn.Module):
    """
    Implementation of the slotGAT slot attention mechanism, where nodes' types correspond to their atomic numbers.

    __init__ Args:
        in_feat: Dimension of input features.
        out_feat: Dimension of output features.
        num_node_types: Number of node types (maximum atomic number).
    """
    def __init__(self,
                 in_feat,
                 out_feat,
                 num_node_types):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_feat, out_feat),
            nn.Tanh()
        )
        self.attn_vector = nn.Parameter(torch.randn(in_feat), requires_grad=True)
        self.softmax = nn.Softmax(dim=1)
        self.num_node_types = num_node_types


    def forward(self, h, num_nodes):
        """
        Aggregate node features, eliminating dimension corresponding to the number of slots (atomic numbers).

        Args:
            h: Node features of size (num_confs, num_nodes, num_node_types, in_feat).
            num_nodes: Number of nodes per conformer graph.

        Returns: Updated node features of size (num_confs, num_nodes, out_feat).
        """

        s = self.mlp(h)  # (num_confs, num_nodes, num_node_types, out_feat)
        print_issue("s", s)

        # Compute attention scores
        b = torch.einsum('cnti,i->cnt', s, self.attn_vector)  # (num_confs, num_nodes, num_node_types)
        print_issue("b", b)
        b = b.sum(dim=1) / num_nodes  # (num_confs, num_node_types)
        print_issue("b after summing", b)
        b = self.softmax(b)
        print_issue("b after softmax", b)

        # Aggregate representations over all slots for each node
        b = b.unsqueeze(1).unsqueeze(-1)  # (num_confs, 1, num_node_types, 1)
        print_issue("b after unsqueeze", b)
        h = torch.sum(b * h, dim=2)  # (num_confs, num_nodes, out_feat)
        print_issue("h updated", h)

        return h

class SlotGATAttention2(nn.Module):
    """
    Implementation of the slotGAT slot attention mechanism, where nodes' types correspond to their conformer indices.

    __init__ Args:
        in_feat: Dimension of input features.
        out_feat: Dimension of output features.
        num_node_types: Number of node types (conformers).
    """
    def __init__(self,
                 in_feat,
                 out_feat,
                 num_node_types):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_feat, out_feat),
            nn.Tanh()
        )
        self.attn_vector = nn.Parameter(torch.randn(in_feat), requires_grad=True)
        self.softmax = nn.Softmax(dim=0)
        self.num_node_types = num_node_types


    def forward(self, h, num_confs):
        """
        Aggregate node features, eliminating dimension corresponding to the number of slots (conformers).

        Args:
            h: Node features of size (num_confs, num_nodes, in_feat).
            num_confs: Number of conformers available for the current molecule.

        Returns: Updated node features of size (num_nodes, out_feat).

        """

        # h size: (num_confs, num_nodes, num_classes)
        s = self.mlp(h)  # (num_confs, num_nodes, in_feat)
        print_issue("s", s)

        # Compute attention scores
        b = torch.einsum('cni,i->cn', s, self.attn_vector)  # (num_confs, num_nodes)
        print_issue("b", b)
        b = b.sum(dim=1) / num_confs  # (num_confs)
        print_issue("b after summing", b)
        b = self.softmax(b)
        print_issue("b after softmax", b)

        # Aggregate representations over all slots for each node
        b = b.unsqueeze(1).unsqueeze(-1)  # (num_confs, 1, 1)
        print_issue("b after unsqueezes", b)
        h = torch.sum(h * b, dim=0)  # (num_nodes, in_feat)
        print_issue("h updated", h)

        return h


class SlotGATTest2(nn.Module):
    """
    Implementation of the SlotGAT model for molecular conformer data. The original paper can be found at
    https://arxiv.org/pdf/2405.01927, and this implementation is based on the original code, which can be found at
    https://github.com/scottjiao/SlotGAT_ICML23/.

    __init__ Args:
        num_parts: Number of parts in the dataset.
        edge_feat: UNUSED.
        num_edge_types: Number of edge types, UNUSED.
        in_dim: Dimension of input features.
        hidden_dim: Dimension latent features.
        num_classes: Dimension of output features for final message passing layer.
        num_layers: Number of message passing layers.
        heads: List containing the number of attention heads for each message passing layer.
        device: Device for tensors to be sent to.
        feat_drop: Dropout probability for message passing layers.
        attn_drop: Dropout probability for slot attention layers.
        negative_slope: Slope for leaky ReLU non-linearity.
        residual: Whether to use residual connection in message passing layers.
        activation: Activation function for message passing layers.
        alpha: Alpha scalar for message passing layers.
        max_atomic_num: Max atomic number for nodes, will be the number of slots in message passing layers.
        max_num_confs: Max number of conformers for molecules, will be the number of slots in slot attention layers.
    """
    def __init__(self,
                 num_parts: int = 1,
                 edge_feat=False,
                 num_edge_types=None,
                 in_dim: int = 10,
                 hidden_dim: int = 64,
                 num_classes: int = 10,
                 num_layers: int = 2,
                 heads: list[int] = [1,1],
                 device: str = f'cuda:{0}',
                 feat_drop: float = 0.,
                 attn_drop: float = 0.,
                 negative_slope: float = 0.2,
                 residual=False,
                 activation=False,
                 alpha: float = 0.,
                 max_atomic_num: int = 100,
                 max_num_confs: int = 3
    ):
        super().__init__()
        self.num_parts = num_parts
        self.edge_feat = edge_feat
        self.num_edge_types = num_edge_types
        self.in_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.num_classes = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.device = device
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.negative_slope = negative_slope
        self.residual = residual
        self.activation = activation
        self.alpha = alpha
        self.max_atomic_num = max_atomic_num
        self.max_num_confs = max_num_confs

        # Initialize GNN for processing node attributes into feature vectors
        self.gnn = slotSchNet(hidden_dim=self.in_dim)

        # Fully connected layers
        self.fc_list = nn.ModuleList(
            [nn.Linear(self.in_dim, self.hidden_dim) for _ in range(self.max_atomic_num)]
        )
        for fc_layer in self.fc_list:
            nn.init.xavier_normal_(fc_layer.weight, gain=1.414)

        # Input projection layer (no residual)
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(
            SlotGATLayer(edge_feat=self.edge_feat,
                         num_edge_types=self.num_edge_types,
                         in_feat=self.hidden_dim,
                         out_feat=self.hidden_dim,
                         num_heads=self.heads[0],
                         feat_drop=self.feat_drop,
                         attn_drop=self.attn_drop,
                         negative_slope=self.negative_slope,
                         residual=self.residual,
                         activation=self.activation,
                         bias=False,
                         alpha=self.alpha,
                         num_node_types=self.max_atomic_num,
                         input_head=False,
                         device=self.device).to(self.device)
        )

        # Hidden layers
        for layer in range(self.num_layers):
            self.gat_layers.append(
                SlotGATLayer(self.edge_feat,
                             self.num_edge_types,
                             self.hidden_dim,  # in_feat
                             self.hidden_dim,  # out_feat
                             self.heads[layer-1],
                             self.feat_drop,
                             self.attn_drop,
                             self.negative_slope,
                             self.residual,
                             self.activation,
                             alpha=self.alpha,
                             num_node_types=self.max_atomic_num,
                             input_head=False,
                             device=self.device).to(self.device)
            )

        # Output projection
        self.gat_layers.append(
            SlotGATLayer(self.edge_feat,
                         self.num_edge_types,
                         in_feat=self.hidden_dim,
                         out_feat=self.num_classes,  # for classification
                         num_heads=self.heads[-1],
                         feat_drop=self.feat_drop,
                         attn_drop=self.attn_drop,
                         negative_slope=self.negative_slope,
                         residual=self.residual,
                         activation=None,
                         alpha=self.alpha,
                         num_node_types=self.max_atomic_num,
                         input_head=False,
                         device=self.device).to(self.device)
        )

        # Slot attention mechanisms
        self.attn_node_types = SlotGATAttention(in_feat=self.hidden_dim,
                                                out_feat=self.hidden_dim,
                                                num_node_types=self.max_atomic_num).to(self.device)
        self.attn_confs = SlotGATAttention2(in_feat=self.hidden_dim,
                                            out_feat=self.num_classes,
                                            num_node_types=self.max_num_confs).to(self.device)

        # Output - linear projection head
        self.proj_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features=self.num_classes, out_features=1)
        )

    def forward(self, data):
        """
        One pass through the model.

        Args:
            data: Dataloader object containing attributes:
                atomic numbers (z)
                node coordinates (pos)
                conformer indices (batch)

        Returns:
            preds: List of predictions of length num_molecules.
        """
        preds = []

        for part_id in range(self.num_parts):
            part_data = data[part_id].to(self.device)

            # Convert data for iteration
            data_list = part_data.to_data_list()

            # Process each molecule separately
            for d in data_list:
                num_confs = len(d.name)

                # Get atomic numbers, node positions, batch
                z, pos, batch = d.x[:, 0], d.pos, d.batch
                num_nodes = d.x.shape[0] // num_confs

                edge_idx = d.edge_index  # To be shared among all conformers

                # Pass conformers through 3D GNN to get node feature vectors
                h_init = self.gnn.forward(z, pos, batch)  # (num_nodes * num_confs, self.in_dim)

                # Slot initialization
                h = torch.zeros_like(h_init)
                for t in range(self.max_atomic_num):
                    # Select nodes with atomic number t
                    mask = (z == int(t))  # (num_nodes * num_confs,)

                    if h_init[mask].shape[0] != 0:
                        # Apply the t-th fully connected layer
                        h[mask] = self.fc_list[t](h_init[mask])  # (-1, self.in_dim)

                # Reformat features for message-passing layers
                masks = torch.zeros(num_confs, num_nodes, self.max_atomic_num, device=h.device)
                # Create mask for slots based on node type

                # Reshape node_type tensor to (num_confs, num_nodes) if it is flat
                if z.dim() == 1:
                    z = z.view(num_confs, num_nodes)

                # Construct mask for each node type
                for t in range(self.max_atomic_num):
                    masks[:, :, t] = (z == t).float()

                # Apply mask
                h = h.view(num_confs, num_nodes, self.in_dim).unsqueeze(2).repeat(1, 1, self.max_atomic_num, 1)
                h = masks.unsqueeze(-1) * h  # (num_confs, num_nodes, num_node_types, in_feat)

                res_attn = None

                # Hidden layers
                for layer in range(self.num_layers):
                    h, res_attn = self.gat_layers[layer](edge_idx=edge_idx,
                                                         node_feats=h,
                                                         edge_attr=0,
                                                         res_attn=res_attn)
                    # (num_confs, num_nodes, layer_out_dim)

                # Output projection
                logits, _ = self.gat_layers[-1](edge_idx, h, edge_attr=0, res_attn=None)
                # ^ (num_confs, num_nodes, num_node_types, layer_out_dim)

                # Attention mechanisms
                # Node type = atomic number
                logits = self.attn_node_types(logits, num_nodes)  # (num_confs, num_nodes, hidden_dim)
                # Node type = conformer index
                logits = self.attn_confs(logits, num_confs)  # (num_nodes, num_classes)

                # Add prediction
                pred = torch.mean(self.proj_head(logits))
                preds.append(pred)

        return torch.stack(preds)







