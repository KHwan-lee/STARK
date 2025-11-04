from .BaseModel import *

class LoraKGE_Layers(BaseModel):
    def __init__(self, args, kg) -> None:
        super(LoraKGE_Layers, self).__init__(args, kg)
        self.lora_ent_embeddings_list = None
        self.lora_rel_embeddings = None

    def store_old_parameters(self):
        for name, param in self.named_parameters():
            name = name.replace('.', '_')
            if param.requires_grad:
                value = param.data
                self.register_buffer(f'old_data_{name}', value.clone().detach())

    def expand_embedding_size(self):
        ent_embeddings = nn.Embedding(self.kg.snapshots[self.args.snapshot + 1].num_ent, self.args.emb_dim).to(self.args.device).double()
        rel_embeddings = nn.Embedding(self.kg.snapshots[self.args.snapshot + 1].num_rel, self.args.emb_dim).to(self.args.device).double()
        xavier_normal_(ent_embeddings.weight)
        xavier_normal_(rel_embeddings.weight)
        return deepcopy(ent_embeddings), deepcopy(rel_embeddings)

    def get_new_ordered_entities(self):
        all_new_entities = {}
        for _ in range(self.kg.snapshots[self.args.snapshot].num_ent, self.kg.snapshots[self.args.snapshot + 1].num_ent):
            #all_new_entities[_] = (0, 0)
            all_new_entities[_] = 0.0
            
        #nodes_ordered_path = f"./data/{self.args.dataset}/{self.args.snapshot + 1}/train_distance_nodes.txt"
        nodes_ordered_path = f"./data/{self.args.dataset}/{self.args.snapshot + 1}/train_distance_nodes2.txt"
        
        with open(nodes_ordered_path, "r", encoding="utf-8") as f:
            lines = list(f.readlines())
            for line in lines:
                line = line.strip()
                line_list = line.split("\t")
                #node, distance, score = int(line_list[0]), int(line_list[1]), float(line_list[2])
                node, score = int(line_list[0]), float(line_list[1])

                if node in all_new_entities:

                    # print(f"Node {node} | Distance: {distance} | Score: {score}")
                    #all_new_entities[node] = (distance, score)
                    all_new_entities[node] = (score)


        #all_new_entities = dict(sorted(all_new_entities.items(), key = lambda kv:(kv[1][0], -kv[1][1])))
        all_new_entities = dict(sorted(all_new_entities.items(), key = lambda kv:(kv[1]), reverse=True))

        self.all_new_entities = all_new_entities
        all_new_entities = list(all_new_entities.keys())


        return all_new_entities

    def expand_lora_embeddings(self):
        self.new_ordered_entities = self.get_new_ordered_entities()
        new_ent_embeddings_len = self.kg.snapshots[self.args.snapshot + 1].num_ent - self.kg.snapshots[self.args.snapshot].num_ent
        print("new entities embeddings length: " + str(new_ent_embeddings_len))
        new_rel_embeddings_len = self.kg.snapshots[self.args.snapshot + 1].num_rel - self.kg.snapshots[self.args.snapshot].num_rel
        print("new relations embeddings length: " + str(new_rel_embeddings_len))
        self.lora_ent_len = (new_ent_embeddings_len + int(self.args.num_ent_layers) - 1) // int(self.args.num_ent_layers)
        print("Each LoRA length: "+ str(self.lora_ent_len))
        tmp_r = self.args.ent_r

        # check
        print(str(self.args.ent_r) + "\n")
        if self.args.explore:
            self.args.ent_r = tmp_r
        print(self.args.using_various_ranks)
        if self.args.using_various_ranks:
            ent_node_list = []
            for k, v in self.all_new_entities.items():
                ent_node_list.append(v)
            self.args.ent_r_list = []
            for i_layer in range(int(self.args.num_ent_layers)):
                self.args.ent_r_list.append(sum(ent_node_list[i_layer * self.lora_ent_len: (i_layer + 1) * self.lora_ent_len]))
            average_nodes = sum(self.args.ent_r_list) / len(self.args.ent_r_list)
            # 0.9 -> 0.25
            r_threshold = int(int(self.args.ent_r) * 0.25)
            self.args.ent_r_list = [int(self.args.ent_r) * i / average_nodes if int(self.args.ent_r) * i / average_nodes > r_threshold else r_threshold for i in self.args.ent_r_list]

            self.args.ent_r_list = [int(i) for i in self.args.ent_r_list]
            self.args.ent_r_list = [int(i) for i in self.args.ent_r_list]
            self.args.ent_r_list = [i if i else 1 for i in self.args.ent_r_list]
            assert len(self.args.ent_r_list) == int(self.args.num_ent_layers)
        elif self.args.using_various_ranks_reverse:
            self.args.ent_r_list = np.linspace(int(self.args.ent_r) // 2, int(self.args.ent_r) // 2 * 3, int(self.args.num_ent_layers)).tolist()
            self.args.ent_r_list = [int(i) for i in self.args.ent_r_list]
            self.args.ent_r_list = [i if i else 1 for i in self.args.ent_r_list]
            self.args.ent_r_list = self.args.ent_r_list[::-1]
            assert len(self.args.ent_r_list) == int(self.args.num_ent_layers)
        else:
            self.args.ent_r_list = [int(self.args.ent_r) // int(self.args.num_ent_layers)] * int(self.args.num_ent_layers)
            self.args.ent_r_list = [i if i else 1 for i in self.args.ent_r_list]
        lora_ent_embeddings_list = []
        for _ in range(int(self.args.num_ent_layers)):
            new_ent_embeddings = loralib.Embedding(self.lora_ent_len, self.args.emb_dim, int(self.args.ent_r_list[_])).to(self.args.device).double()
            xavier_normal_(new_ent_embeddings.weight)
            lora_ent_embeddings_list.append(deepcopy(new_ent_embeddings))
        new_rel_embeddings = loralib.Embedding(new_rel_embeddings_len, self.args.emb_dim, int(self.args.rel_r)).to(self.args.device).double()
        xavier_normal_(new_rel_embeddings.weight)
        return deepcopy(lora_ent_embeddings_list), deepcopy(new_rel_embeddings)

    def switch_snapshot(self):
        if self.lora_ent_embeddings_list != None:
            new_ent_embeddings = self.ent_embeddings.weight.data
            new_rel_embeddings = self.rel_embeddings.weight.data
            for lora_id in range(int(self.args.num_ent_layers) - 1):

                start_id = self.kg.snapshots[self.args.snapshot - 1].num_ent + lora_id * self.lora_ent_len
                new_ent_embeddings[start_id: start_id + self.lora_ent_len] = Parameter(self.lora_ent_embeddings_list[lora_id].forward(torch.arange(self.lora_ent_len).to(self.args.device)))
                print("debugging: " + str(start_id) + " " + str(self.lora_ent_len))
            last_start_id = self.kg.snapshots[self.args.snapshot - 1].num_ent + (int(self.args.num_ent_layers) - 1) * self.lora_ent_len
            last_lora_id = int(self.args.num_ent_layers) - 1
            new_ent_embeddings[last_start_id:] = Parameter(self.lora_ent_embeddings_list[last_lora_id].forward(torch.arange(len(new_ent_embeddings[last_start_id:])).to(self.args.device)))
            ent_indices = list(range(self.kg.snapshots[self.args.snapshot - 1].num_ent)) + self.new_ordered_entities
            print(str(len(new_ent_embeddings)) + " " + str(len(ent_indices)))
            assert len(new_ent_embeddings) == len(ent_indices)
            new_ent_embeddings = new_ent_embeddings[ent_indices]
            new_rel_embeddings[self.kg.snapshots[self.args.snapshot - 1].num_rel:] = Parameter(deepcopy(self.lora_rel_embeddings.forward(torch.arange(len(self.lora_rel_embeddings.weight)).to(self.args.device))))       
            self.ent_embeddings.weight = Parameter(new_ent_embeddings)
            self.rel_embeddings.weight = Parameter(new_rel_embeddings)

        if self.args.snapshot != 4:
            self.store_old_parameters()


            ent_embeddings, rel_embeddings = self.expand_embedding_size()

            new_ent_embeddings = ent_embeddings.weight.data
            new_rel_embeddings = rel_embeddings.weight.data
            new_ent_embeddings[:self.kg.snapshots[self.args.snapshot].num_ent] = Parameter(
                self.ent_embeddings.weight.data
            )
            new_rel_embeddings[:self.kg.snapshots[self.args.snapshot].num_rel] = Parameter(
                self.rel_embeddings.weight.data
            )
            self.ent_embeddings.weight = Parameter(new_ent_embeddings)
            self.rel_embeddings.weight = Parameter(new_rel_embeddings)
            self.ent_embeddings.requires_grad = False
            self.rel_embeddings.requires_grad = False

            self.lora_ent_embeddings_list_tmp, self.lora_rel_embeddings = self.expand_lora_embeddings()
            self.lora_ent_embeddings_list = nn.ModuleList(self.lora_ent_embeddings_list_tmp)

class TransE(LoraKGE_Layers):
    def __init__(self, args, kg) -> None:
        super(TransE, self).__init__(args, kg)
        self.huber_loss = torch.nn.HuberLoss(reduction='sum')

    def new_loss(self, head, rel, tail=None, label=None):
        """ return loss of new facts """
        return self.margin_loss(head, rel, tail, label) / head.size(0)

    def score_fun(self, h, r, t):
        """ Score function: L1-norm (h + r - t) """
        h = self.norm_ent(h)
        r = self.norm_rel(r)
        t = self.norm_ent(t)
        return torch.norm(h + r - t, 1, -1)

    def split_pn_score(self, score, label):
        """
        split postive triples and negtive triples
        :param score: scores of all facts
        :param label: postive facts: 1, negtive facts: -1
        """
        p_score = score[torch.where(label > 0)]
        n_score = (score[torch.where(label < 0)]).reshape(-1, self.args.neg_ratio).mean(dim=1)
        return p_score, n_score

    def get_lora_embeddings(self):
        lora_ent_embeddings = self.lora_ent_embeddings_list[0].forward(torch.arange(self.lora_ent_len).to(self.args.device))
        for lora_id in range(1, int(self.args.num_ent_layers)):
            lora_ent_embeddings = torch.cat((lora_ent_embeddings, self.lora_ent_embeddings_list[lora_id].forward(torch.arange(self.lora_ent_len).to(self.args.device))), dim=0)
        lora_rel_embeddings = self.lora_rel_embeddings.forward(torch.arange(len(self.lora_rel_embeddings.weight)).to(self.args.device))
        return lora_ent_embeddings, lora_rel_embeddings

    def embedding(self, stage=None):
        '''get embeddings without lora embeddings'''
        if self.args.snapshot == 0:
            ent_embeddings = self.ent_embeddings.weight
            rel_embeddings = self.rel_embeddings.weight
        else:
            ent_embeddings = self.old_data_ent_embeddings_weight
            rel_embeddings = self.old_data_rel_embeddings_weight
        return ent_embeddings, rel_embeddings


    def predict(self, head, relation, stage='Valid'):
        """ Score all candidate facts for evaluation """
        if stage != 'Test':
            num_ent = self.kg.snapshots[self.args.snapshot_valid].num_ent
        else:
            num_ent = self.kg.snapshots[self.args.snapshot_test].num_ent
        if self.args.snapshot == 0:
            ent_embeddings, rel_embeddings = self.embedding(stage)
            h = torch.index_select(ent_embeddings, 0, head)
            r = torch.index_select(rel_embeddings, 0, relation)
            t_all = ent_embeddings[:num_ent]
        else:
            ent_embeddings, rel_embeddings = self.embedding(stage)
            lora_ent_embeddings, lora_rel_embeddings = self.get_lora_embeddings()
            all_ent_embeddings = torch.cat([ent_embeddings, lora_ent_embeddings], dim=0)
            all_rel_embeddings = torch.cat([rel_embeddings, lora_rel_embeddings], dim=0)
            ent_indices = list(range(self.kg.snapshots[self.args.snapshot - 1].num_ent)) + self.new_ordered_entities
            all_ent_embeddings = all_ent_embeddings[ent_indices]
            h = torch.index_select(all_ent_embeddings, 0, head)
            r = torch.index_select(all_rel_embeddings, 0, relation)
            t_all = all_ent_embeddings[:num_ent]

        h = self.norm_ent(h)
        r = self.norm_rel(r)
        t_all = self.norm_ent(t_all)

        """ h + r - t """
        pred_t = h + r
        score = 9.0 - torch.norm(pred_t.unsqueeze(1) - t_all, p=1, dim=2)
        score = torch.sigmoid(score)
        return score

    # def margin_loss(self, head, rel, tail, label=None):
    #     """ Pair wise margin loss: L1-norm (h + r - t) """
    #     if self.args.snapshot == 0:
    #         ent_embeddings, rel_embeddings = self.embedding('Train')
    #         h = torch.index_select(ent_embeddings, 0, head)
    #         r = torch.index_select(rel_embeddings, 0, rel)
    #         t = torch.index_select(ent_embeddings, 0, tail)
    #     else:
    #         ent_embeddings, rel_embeddings = self.embedding('Train')
    #         lora_ent_embeddings, lora_rel_embeddings = self.get_lora_embeddings()
    #         all_ent_embeddings = torch.cat([ent_embeddings, lora_ent_embeddings], dim=0)
    #         all_rel_embeddings = torch.cat([rel_embeddings, lora_rel_embeddings], dim=0)
    #         ent_indices = list(range(self.kg.snapshots[self.args.snapshot - 1].num_ent)) + self.new_ordered_entities
    #         all_ent_embeddings = all_ent_embeddings[ent_indices]
    #         h = torch.index_select(all_ent_embeddings, 0, head)
    #         r = torch.index_select(all_rel_embeddings, 0, rel)
    #         t = torch.index_select(all_ent_embeddings, 0, tail)
    #     # score function for forward propagation
    #     score = self.score_fun(h, r, t)
    #     p_score, n_score = self.split_pn_score(score, label)
    #     y = torch.Tensor([-1]).to(self.args.device)
    #     L_rank =  self.margin_loss_func(p_score, n_score, y)
    #     # for original, pgbtw
    #     return L_rank
    #     # y = torch.Tensor([-1]).to(self.args.device)
    #     # L_pos =  F.relu(p_score - 8.0).mean()
    #     # L_neg = F.relu(4.0 - n_score).mean()
    #     # loss = L_pos + L_neg
    #     # print(L_rank.item(), L_s.item())
    #     # for original, pgbtw
    #     #return loss
        
    def margin_loss(self, head, rel, tail, label=None):
        """ Pair wise margin loss: L1-norm (h + r - t) """
        if self.args.snapshot == 0:
            ent_embeddings, rel_embeddings = self.embedding('Train')
            h = torch.index_select(ent_embeddings, 0, head)
            r = torch.index_select(rel_embeddings, 0, rel)
            t = torch.index_select(ent_embeddings, 0, tail)
        else:
            ent_embeddings, rel_embeddings = self.embedding('Train')
            lora_ent_embeddings, lora_rel_embeddings = self.get_lora_embeddings()
            all_ent_embeddings = torch.cat([ent_embeddings, lora_ent_embeddings], dim=0)
            all_rel_embeddings = torch.cat([rel_embeddings, lora_rel_embeddings], dim=0)
            ent_indices = list(range(self.kg.snapshots[self.args.snapshot - 1].num_ent)) + self.new_ordered_entities
            all_ent_embeddings = all_ent_embeddings[ent_indices]
            h = torch.index_select(all_ent_embeddings, 0, head)
            r = torch.index_select(all_rel_embeddings, 0, rel)
            t = torch.index_select(all_ent_embeddings, 0, tail)

        # score function for forward propagation
        score = self.score_fun(h, r, t)
        p_score, n_score = self.split_pn_score(score, label)
        y = torch.Tensor([-1]).to(self.args.device)
        L_rank =  self.margin_loss_func(p_score, n_score, y)
        #for original, pgbtw
        #return L_rank
        

        mask      = (label > 0)
        head_pos  = head[mask]
        rel_pos   = rel[mask]


        with torch.no_grad():
            counts = torch.tensor(
                [len(self.kg.train_hr2t[(int(h_),int(r_))])
                 for h_, r_ in zip(head_pos, rel_pos)],
                device=score.device,
                dtype=score.dtype
            ) 


        lambda_dyn = float(getattr(self.args, "lambda_dyn", 1.0))
        margins = lambda_dyn * counts.sqrt()                 # (P,)


        avg_margin = (lambda_dyn * counts.sqrt()).mean().item()
        avg_p     = p_score.mean().item()


        L_hub = torch.relu(p_score - margins).sum()    # 또는 .mean()
        # L_hub = L_hub / p_score.size(0)


        snap_i = int(self.args.snapshot)
        w = float(self.args.fusion_weight)
        if snap_i > 0:
            ent_prev = self.kg.snapshots[snap_i - 1].num_ent
            tri_prev = len(self.kg.snapshots[snap_i - 1].train_all)
            ent_cur    = len(self.new_ordered_entities)
            tri_cur = len(self.kg.snapshots[snap_i].train)
            
            ent_total = ent_prev + ent_cur
            tri_total = tri_prev + tri_cur
            eta   = w * (ent_prev / ent_total) + (1 - w) * (tri_cur / tri_total)
        else:
            eta = 0.1


        # L_rank = L_rank * alpha
        hub_scale = float(self.args.hub_scale)
        # L_hub = L_hub * hub_scale
        L_hub = L_hub * eta * hub_scale

        # print(f"L_rank={L_rank.item():.4f}, L_hub={L_hub.item():.4f}")
        total_loss = L_rank + L_hub
        return total_loss
    
    # def margin_loss(self, head, rel, tail, label=None):
    #     """ Pair wise margin loss: L1-norm (h + r - t) """
    #     if self.args.snapshot == 0:
    #         ent_embeddings, rel_embeddings = self.embedding('Train')
    #         h = torch.index_select(ent_embeddings, 0, head)
    #         r = torch.index_select(rel_embeddings, 0, rel)
    #         t = torch.index_select(ent_embeddings, 0, tail)
    #     else:
    #         ent_embeddings, rel_embeddings = self.embedding('Train')
    #         lora_ent_embeddings, lora_rel_embeddings = self.get_lora_embeddings()
    #         all_ent_embeddings = torch.cat([ent_embeddings, lora_ent_embeddings], dim=0)
    #         all_rel_embeddings = torch.cat([rel_embeddings, lora_rel_embeddings], dim=0)
    #         ent_indices = list(range(self.kg.snapshots[self.args.snapshot - 1].num_ent)) + self.new_ordered_entities
    #         all_ent_embeddings = all_ent_embeddings[ent_indices]
    #         h = torch.index_select(all_ent_embeddings, 0, head)
    #         r = torch.index_select(all_rel_embeddings, 0, rel)
    #         t = torch.index_select(all_ent_embeddings, 0, tail)
        
    #     # score function for forward propagation
    #     if label is not None:
    #         mask = (label > 0)
    #         h, r, t = h[mask], r[mask], t[mask]
        
    #     score = self.score_fun(h, r, t)  # shape: (P,)

    #     with torch.no_grad():
    #         counts = torch.tensor(
    #             [len(self.kg.train_hr2t[(int(h_), int(r_))]) for h_, r_ in zip(head[mask], rel[mask])],
    #             device=score.device,
    #             dtype=score.dtype
    #         )  # shape: (P,)

    #     gamma   = float(self.args.margin2)
    #     margins = gamma * counts.sqrt()    # shape: (P,)

    #     loss = torch.relu(score - margins).sum()
    #     loss = loss / score.size(0)

    #     return loss
    

    def get_TransE_loss(self, head, relation, tail, label):
        return self.new_loss(head, relation, tail, label)

    def loss(self, head, relation, tail=None, label=None):
        loss = self.get_TransE_loss(head, relation, tail, label)
        return loss