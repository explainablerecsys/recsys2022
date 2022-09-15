from __future__ import absolute_import, division, print_function

import torch
from torch import nn
from torch.nn import functional as F

from my_knowledge_graph import *
from cafe_utils import *


class EntityEmbeddingModel(nn.Module):
    def __init__(self, entity_info, embed_size, init_embed=None):
        super(EntityEmbeddingModel, self).__init__()
        self.entity_info = entity_info
        self.embed_size = embed_size
        # initialize embedding
        for name in entity_info:
            info = entity_info[name]
            embed = nn.Embedding(info["vocab_size"] + 1, self.embed_size, padding_idx=-1, sparse=False)
            initrange = 0.5 / self.embed_size
            weight = torch.FloatTensor(info["vocab_size"] + 1, self.embed_size).uniform_(-initrange, initrange)
            embed.weight = nn.Parameter(weight)
            setattr(self, name, embed)

        if init_embed is not None:
            for name in entity_info:
                weight = torch.from_numpy(init_embed[name])
                getattr(self, name).data = weight

    def forward(self, entity, ids=None):
        if ids is None:
            return getattr(self, entity).weight
        return getattr(self, entity)(ids)

    def vocab_size(self, entity):
        return self.entity_info[entity]["vocab_size"] + 1


class RelationModule(nn.Module):
    def __init__(self, embed_size, relation_info):
        super(RelationModule, self).__init__()
        self.name = relation_info["name"]
        self.eh_name = relation_info["entity_head"]
        self.et_name = relation_info["entity_tail"]
        self.fc1 = nn.Linear(embed_size * 2, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, embed_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs):
        """Compute log probability of output entity.
        Args:
            x: a FloatTensor of size [bs, input_size].
        Returns:
            FloatTensor of log probability of size [bs, output_size].
        """
        eh_vec, user_vec = inputs
        x = torch.cat([eh_vec, user_vec], dim=-1)
        x = self.bn1(self.dropout(F.relu(self.fc1(x))))
        out = self.fc2(x) + eh_vec
        return out


class DeepRelationModule(nn.Module):
    def __init__(self, embed_size, relation_info, use_dropout=True):
        super(DeepRelationModule, self).__init__()
        self.name = relation_info["name"]
        self.eh_name = relation_info["entity_head"]
        self.et_name = relation_info["entity_tail"]
        input_size = embed_size * 2
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, input_size)
        self.bn2 = nn.BatchNorm1d(input_size)
        self.fc3 = nn.Linear(input_size, embed_size)
        if use_dropout:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = None

    def forward(self, inputs):
        eh_vec, user_vec = inputs
        feature = torch.cat([eh_vec, user_vec], dim=-1)
        x = F.relu(self.fc1(feature))
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.bn1(x)
        x = F.relu(self.fc2(x) + feature)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.bn2(x)
        out = self.fc3(x)
        return out


class SymbolicNetwork(nn.Module):
    def __init__(self, relation_info, embedding, deep_module, use_dropout, device):
        """Initialize the network.
        Args:
            entity_info: a dict whose key is entity name and value contains attributes.
            relation_info:
            embed_size: embedding size.
        """
        super(SymbolicNetwork, self).__init__()
        self.embedding = embedding
        self.embed_size = embedding.embed_size
        self.device = device

        self._create_modules(relation_info, deep_module, use_dropout)
        self.nll_criterion = nn.NLLLoss(reduction="none")
        self.ce_loss = nn.CrossEntropyLoss()

    def _create_modules(self, relation_info, use_deep=False, use_dropout=True):
        """Create module for each relation."""
        for name in relation_info:
            info = relation_info[name]
            if not use_deep:
                module = RelationModule(self.embed_size, info)
            else:
                module = DeepRelationModule(self.embed_size, info, use_dropout)
            setattr(self, name, module)

    def _get_modules(self, metapath):
        """Get list of modules by metapath."""
        module_seq = []  # seq len = len(metapath)-1
        for relation, _ in metapath[1:]:
            module = getattr(self, relation)
            module_seq.append(module)
        return module_seq

    def _forward(self, modules, uids):
        outputs = []
        batch_size = uids.size(0)
        user_vec = self.embedding(USER, uids)  # [bs, d]
        input_vec = user_vec
        for module in modules:
            out = module((input_vec, user_vec)).view(batch_size, -1)  # [bs, d]
            outputs.append(out)
            input_vec = out
        return outputs

    def forward(self, metapath, pos_paths, neg_pids):
        """Compute loss.
        Args:
            metapath: list of relations, e.g. [USER, (r1, e1),..., (r_n, e_n)].
            uid: a LongTensor of user ids, with size [bs, ].
            target_path: a LongTensor of node ids, with size [bs, len(metapath)],
                    e.g. each path contains [u, e1,..., e_n].
            indicator: an integer value indicating good/bad path.
            teacher_forcing: use teacher forcing or not.
        Returns:
            logprobs: sum of log probabilities of given target node ids, with size [bs, ].
        """
        # Note: len(modules) = len(metapath)-1 = len(target_path)-1
        modules = self._get_modules(metapath)
        outputs = self._forward(modules, pos_paths[:, 0])

        # Path regularization loss
        reg_loss = 0
        scores = 0
        for i, module in enumerate(modules):
            et_vecs = self.embedding(module.et_name)
            scores = torch.matmul(outputs[i], et_vecs.t())
            reg_loss += self.ce_loss(scores, pos_paths[:, i + 1])

        # Ranking loss
        logprobs = F.log_softmax(scores, dim=1)  # [bs, vocab_size]
        # predict = outputs[-1]  # [bs, d]
        # pos_products = self.embedding(PRODUCT, pos_paths[:, -1])  # [bs, d]
        # pos_score = torch.sum(predict * pos_products, dim=1)  # [bs, ]
        # neg_products = self.embedding(PRODUCT, neg_pids)  # [bs, d]
        # neg_score = torch.sum(predict * neg_products, dim=1)  # [bs, ]
        pos_score = torch.gather(logprobs, 1, pos_paths[:, -1].view(-1, 1))
        neg_score = torch.gather(logprobs, 1, neg_pids.view(-1, 1))
        rank_loss = torch.sigmoid(neg_score - pos_score).mean()

        return reg_loss, rank_loss

    def forward_simple(self, metapath, uids, pids):
        modules = self._get_modules(metapath)
        outputs = self._forward(modules, uids)

        # Path regularization loss
        products = self.embedding(PRODUCT)  # [bs, d]
        scores = torch.matmul(outputs[-1], products.t())  # [bs, vocab_size]
        logprobs = F.log_softmax(scores, dim=1)  # [bs, vocab_size]
        pid_logprobs = logprobs.gather(1, pids.view(-1, 1)).view(-1)
        return pid_logprobs

    def infer_direct(self, metapath, uid, pids):
        if len(pids) == 0:
            return []
        modules = self._get_modules(metapath)
        uid_tensor = torch.LongTensor([uid]).to(self.device)
        outputs = self._forward(modules, uid_tensor)  # list of tensor of [1, d]

        # Path regularization loss
        pids_tensor = torch.LongTensor(pids).to(self.device)
        products = self.embedding(PRODUCT)  # [bs, d]
        scores = torch.matmul(outputs[-1], products.t())  # [1, vocab_size]
        logprobs = F.log_softmax(scores, dim=1)  # [1, vocab_size]
        pid_logprobs = logprobs[0][pids_tensor]
        return pid_logprobs.detach().cpu().numpy().tolist()

    def infer_with_path(self, metapath, uid, kg_mask, excluded_pids=None, topk_paths=10):
        """Reasoning paths over kg."""
        modules = self._get_modules(metapath)
        uid_tensor = torch.LongTensor([uid]).to(self.device)
        outputs = self._forward(modules, uid_tensor)  # list of tensor of [1, d]

        layer_logprobs = []
        for i, module in enumerate(modules):
            et_vecs = self.embedding(module.et_name)
            scores = torch.matmul(outputs[i], et_vecs.t())  # [1, vocab_size]
            logprobs = F.log_softmax(scores[0], dim=0)  # [vocab_size, ]
            layer_logprobs.append(logprobs)

        # Decide adaptive sampling size.
        num_valid_ids = len(kg_mask.get_ids(USER, uid, modules[0].name))
        if num_valid_ids <= 0:
            return []
        if topk_paths <= num_valid_ids:
            sample_sizes = [topk_paths, 1, 1]
        else:
            sample_sizes = [num_valid_ids, int(topk_paths / num_valid_ids) + 1, 1]

        result_paths = [([uid], [])]  # (list of ids, list of scores)
        for i, module in enumerate(modules):  # iterate over each level
            # If remove excluded item for the last node.
            # if i == len(modules) - 1 and excluded_pids is not None:
            #    excluded_pids = torch.LongTensor(excluded_pids).to(self.device)
            #    layer_logprobs[i][excluded_pids] = -9999

            tmp_paths = []
            visited_ids = []
            for path, value in result_paths:  # both are lists
                # Find valid node ids that are unvisited and not excluded pids.
                valid_et_ids = kg_mask.get_ids(module.eh_name, path[-1], module.name)
                valid_et_ids = set(valid_et_ids).difference(visited_ids)
                if i == len(modules) - 1 and excluded_pids is not None:
                    valid_et_ids = valid_et_ids.difference(excluded_pids)
                if len(valid_et_ids) <= 0:
                    continue
                valid_et_ids = list(valid_et_ids)

                # Compute top k nodes.
                valid_et_ids = torch.LongTensor(valid_et_ids).to(self.device)
                valid_et_logprobs = layer_logprobs[i].index_select(0, valid_et_ids)
                k = min(sample_sizes[i], len(valid_et_ids))
                topk_et_logprobs, topk_idxs = valid_et_logprobs.topk(k)
                topk_et_ids = valid_et_ids.index_select(0, topk_idxs)
                # layer_logprobs[i][topk_et_ids] = -9999  # prevent the nodes being selected again

                # Add nodes to path separately.
                topk_et_ids = topk_et_ids.detach().cpu().numpy()
                topk_et_logprobs = topk_et_logprobs.detach().cpu().numpy()
                for j in range(topk_et_ids.shape[0]):
                    new_path = path + [topk_et_ids[j]]
                    new_value = value + [topk_et_logprobs[j]]
                    tmp_paths.append((new_path, new_value))
                    # Remember to add the node to visited list!!!
                    visited_ids.append(topk_et_ids[j])

            if len(tmp_paths) <= 0:
                return []
            result_paths = tmp_paths

        return result_paths


def create_symbolic_model(args, kg, train=True, pretrain_embeds=None):
    """Create neural symbolic model based on KG.

    Args:
        args: arguments.
        kg (MyKnowledgeGraph): KG object.
        train (bool, optional): is training model. Defaults to True.

    Returns:
        SymbolicNetwork: model object.
    """
    entity_info, relation_info = {}, {}
    for entity in kg.G:
        entity_info[entity] = {"vocab_size": len(kg.G[entity])}
    for rel in kg.relation_info:
        relation_info[rel] = {
            "name": rel,
            "entity_head": kg.relation_info[rel][0],
            "entity_tail": kg.relation_info[rel][1],
        }

    # pretrain_embeds = utils.load_embed(args.dataset) if train else None
    entity_embed_model = EntityEmbeddingModel(entity_info, args.embed_size, init_embed=pretrain_embeds)
    model = SymbolicNetwork(relation_info, entity_embed_model, args.deep_module, args.use_dropout, args.device)
    model = model.to(args.device)
    if train:
        model.train()
    else:
        assert hasattr(args, "symbolic_model")
        print("Load symbolic model:", args.symbolic_model)
        pretrain_sd = torch.load(args.symbolic_model, map_location=lambda storage, loc: storage)
        model_sd = model.state_dict()
        model_sd.update(pretrain_sd)
        model.load_state_dict(model_sd)
        model.eval()

    return model
