from .BaseModel import *

class LoraKGE_Layers(BaseModel):
    def __init__(self, args, kg) -> None:
        super(LoraKGE_Layers, self).__init__(args, kg)
        self.lora_ent_embeddings_list = None
        self.lora_edge_embeddings_list = None

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
            all_new_entities[_] = (0, 0)
        nodes_ordered_path = f"./data/{self.args.dataset}/{self.args.snapshot + 1}/train_distance_nodes2.txt"
        with open(nodes_ordered_path, "r", encoding="utf-8") as f:
            lines = list(f.readlines())
            for line in lines:
                line = line.strip()
                line_list = line.split("\t")
                node, distance, score = int(line_list[0]), int(line_list[1]), float(line_list[2])
                if node in all_new_entities:
                    # 추가된 디버그 출력
                    # print(f"Node {node} | Distance: {distance} | Score: {score}") 
                    all_new_entities[node] = (distance, score)
        # 수정한 부분
        #all_new_entities = dict(sorted(all_new_entities.items(), key = lambda kv:(kv[1][0], kv[1][1])))
        all_new_entities = dict(sorted(all_new_entities.items(), key = lambda kv:(kv[1][1]), reverse=True))
        
        self.all_new_entities = all_new_entities
        all_new_entities = list(all_new_entities.keys())

        # 디버깅용 출력: 정렬 후의 값 확인
        # for node, (distance, score) in self.all_new_entities.items():
          #  print(f"Sorted Node {node} | Distance: {distance} | Score: {score}")
        
        return all_new_entities

    def get_new_ordered_edges(self):
        """
        Load and sort edges based on edge betweenness centrality.
        """
        all_new_edges = {}
        for _ in range(self.kg.snapshots[self.args.snapshot].num_rel, self.kg.snapshots[self.args.snapshot + 1].num_rel):
            all_new_edges[_] = (0)
        # 간선 중심성 데이터를 로드
        edge_betweenness_path = f"./data/{self.args.dataset}/{self.args.snapshot + 1}/train_relation_betweenness.txt"

        with open(edge_betweenness_path, "r", encoding="utf-8") as f:
            lines = list(f.readlines())
            for line in lines:
                line_list = line.strip().split("\t")
                rel, betweenness = int(line_list[0]), float(line_list[1])
                if rel in all_new_edges:
                    all_new_edges[rel] = (betweenness)

        # 중심성 값 기준으로 간선을 정렬
        all_new_edges = dict(sorted(all_new_edges.items(), key=lambda kv: (kv[1]), reverse=True))
        self.all_new_edges = all_new_edges
        all_new_edges = list(all_new_edges.keys())
        
        #self.new_ordered_edges = [edge for edge, _ in sorted_edges]
        # 디버깅 출력: 정렬된 간선 정보 확인
        #for edge, centrality in sorted_edges[:10]:  # 상위 10개 출력
            # print(f"Edge {edge} | Betweenness: {centrality}")

        return all_new_edges

    def expand_lora_embeddings(self):
        self.new_ordered_entities = self.get_new_ordered_entities()
        # print(f"Debug: self.new_ordered_entities = {self.new_ordered_entities}")
        new_ent_embeddings_len = self.kg.snapshots[self.args.snapshot + 1].num_ent - self.kg.snapshots[self.args.snapshot].num_ent
        
        self.lora_ent_len = (new_ent_embeddings_len + int(self.args.num_ent_layers) - 1) // int(self.args.num_ent_layers)
        tmp_r = self.args.ent_r
        # 이것도 빼야 맞는듯
        # self.args.ent_r = (self.lora_ent_len // 20) if (self.lora_ent_len // 20) > int(self.args.ent_r) else self.args.ent_r

        # check
        print(str(self.args.ent_r) + "\n")
        if self.args.explore:
            self.args.ent_r = tmp_r
        print(self.args.using_various_ranks)
        if self.args.using_various_ranks:
            ent_node_list = []
            for k, v in self.all_new_entities.items():
                ent_node_list.append(v[1])
            self.args.ent_r_list = []
            for i_layer in range(int(self.args.num_ent_layers)):
                self.args.ent_r_list.append(sum(ent_node_list[i_layer * self.lora_ent_len: (i_layer + 1) * self.lora_ent_len]))
            average_nodes = sum(self.args.ent_r_list) / len(self.args.ent_r_list)
            # 이거 0.9로 잡은건 너무 주작인거 같음..
            r_threshold = int(int(self.args.ent_r) * 0.25)
            self.args.ent_r_list = [int(self.args.ent_r) * i / average_nodes if int(self.args.ent_r) * i / average_nodes > r_threshold else r_threshold for i in self.args.ent_r_list]
            # 이건 왜 2번 반복되는가? -> 실수라고 함
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

        # Step 2: Load and process new edges (relations)
        self.new_ordered_edges = self.get_new_ordered_edges()
        # print(f"Debug: self.new_ordered_edges = {self.new_ordered_edges}")
        # new_edge_embeddings_len = len(self.new_ordered_edges)
        new_edge_embeddings_len = self.kg.snapshots[self.args.snapshot + 1].num_rel - self.kg.snapshots[self.args.snapshot].num_rel
        self.lora_edge_len = (new_edge_embeddings_len + int(self.args.num_rel_layers) - 1) // int(self.args.num_rel_layers)

        if self.args.using_various_ranks:
            edge_centrality_list = []
            for edge, betweenness in self.all_new_edges.items():
                edge_centrality_list.append(betweenness)
            self.args.edge_r_list = []
            for i_layer in range(int(self.args.num_rel_layers)):
                group_centrality = sum(edge_centrality_list[i_layer * self.lora_edge_len: (i_layer + 1) * self.lora_edge_len])
                total_centrality = sum(edge_centrality_list)
                calculated_rank = int(int(self.args.rel_r) * group_centrality / total_centrality) if total_centrality > 0 else int(self.args.rel_r)
                threshold = int(int(self.args.rel_r) * 0.25)  # Set threshold as 25% of rel_r
                self.args.edge_r_list.append(max(calculated_rank, threshold))
        elif self.args.using_various_ranks_reverse:
            self.args.edge_r_list = np.linspace(self.args.rel_r // 2, self.args.rel_r * 1.5, int(self.args.num_rel_layers)).tolist()
            self.args.edge_r_list = [max(int(i), 1) for i in self.args.edge_r_list[::-1]]
        else:
            self.args.edge_r_list = [max(int(int(self.args.rel_r) // int(self.args.num_rel_layers)), 1)] * int(self.args.num_rel_layers)

        lora_edge_embeddings_list = []
        for i_layer in range(int(self.args.num_rel_layers)):
            new_edge_embeddings = loralib.Embedding(self.lora_edge_len, self.args.emb_dim, int(self.args.edge_r_list[i_layer])).to(self.args.device).double()
            xavier_normal_(new_edge_embeddings.weight)
            lora_edge_embeddings_list.append(deepcopy(new_edge_embeddings))
        
        # new_rel_embeddings = loralib.Embedding(new_rel_embeddings_len, self.args.emb_dim, int(self.args.rel_r)).to(self.args.device).double()
        # xavier_normal_(new_rel_embeddings.weight)

        self.lora_ent_embeddings_list = nn.ModuleList(lora_ent_embeddings_list)
        self.lora_edge_embeddings_list = nn.ModuleList(lora_edge_embeddings_list)
        #return self.lora_ent_embeddings_list, self.lora_edge_embeddings_list
        return deepcopy(lora_ent_embeddings_list), deepcopy(lora_edge_embeddings_list)



    def switch_snapshot(self):
        if self.lora_ent_embeddings_list is not None:
            new_ent_embeddings = self.ent_embeddings.weight.data

            # Update entity embeddings with LoRA
            for lora_id in range(int(self.args.num_ent_layers) - 1):
                start_id = self.kg.snapshots[self.args.snapshot - 1].num_ent + lora_id * self.lora_ent_len
                print("debugging: " + str(start_id) + " " + str(self.lora_ent_len))
                new_ent_embeddings[start_id: start_id + self.lora_ent_len] = Parameter(
                    self.lora_ent_embeddings_list[lora_id].forward(torch.arange(self.lora_ent_len).to(self.args.device))
                )
            
            # Handle the last group of entities
            last_start_id = self.kg.snapshots[self.args.snapshot - 1].num_ent + (int(self.args.num_ent_layers) - 1) * self.lora_ent_len
            last_lora_id = int(self.args.num_ent_layers) - 1
            new_ent_embeddings[last_start_id:] = Parameter(
                self.lora_ent_embeddings_list[last_lora_id].forward(torch.arange(len(new_ent_embeddings[last_start_id:])).to(self.args.device))
            )

            # Adjust entity indices for the current snapshot
            ent_indices = list(range(self.kg.snapshots[self.args.snapshot - 1].num_ent)) + self.new_ordered_entities
            # print(str(len(new_ent_embeddings)) + " " + str(len(ent_indices)))
            assert len(new_ent_embeddings) == len(ent_indices)
            new_ent_embeddings = new_ent_embeddings[ent_indices]

            # Update relation embeddings with LoRA
            new_rel_embeddings = self.rel_embeddings.weight.data
            # print(f"self.lora_edge_embeddings_list length: {len(self.lora_edge_embeddings_list)}")
            # print(f"new_rel_embeddings.size(0): {new_rel_embeddings.size(0)}")
            for lora_id in range(int(self.args.num_rel_layers) - 1):
                start_id = self.kg.snapshots[self.args.snapshot - 1].num_rel + lora_id * self.lora_edge_len
                print("debugging: " + str(start_id) + " " + str(self.lora_edge_len))
                # print(f"self.lora_edge_len: {(self.lora_edge_len)}")
                new_rel_embeddings[start_id: start_id + self.lora_edge_len] = Parameter(
                    self.lora_edge_embeddings_list[lora_id].forward(torch.arange(self.lora_edge_len).to(self.args.device))
                )
            
            # Handle the last group of relations
            last_start_id = self.kg.snapshots[self.args.snapshot - 1].num_rel + (int(self.args.num_rel_layers) - 1) * self.lora_edge_len
            last_lora_id = int(self.args.num_rel_layers) - 1
            new_rel_embeddings[last_start_id:] = Parameter(
                self.lora_edge_embeddings_list[last_lora_id].forward(torch.arange(len(new_rel_embeddings[last_start_id:])).to(self.args.device))
            )

            # Adjust relation indices for the current snapshot
            rel_indices = list(range(self.kg.snapshots[self.args.snapshot - 1].num_rel)) + self.new_ordered_edges
            # print(str(len(new_rel_embeddings)) + " " + str(len(rel_indices)))
            assert len(new_rel_embeddings) == len(rel_indices)
            new_rel_embeddings = new_rel_embeddings[rel_indices]

            # Assign updated embeddings
            self.ent_embeddings.weight = Parameter(new_ent_embeddings)
            self.rel_embeddings.weight = Parameter(new_rel_embeddings)

        # Store current parameters for backward consistency
        self.store_old_parameters()

        # Expand embedding sizes for the next snapshot
        ent_embeddings, rel_embeddings = self.expand_embedding_size()
        new_ent_embeddings = ent_embeddings.weight.data
        new_rel_embeddings = rel_embeddings.weight.data

        # Copy existing embeddings to the expanded embeddings
        new_ent_embeddings[:self.kg.snapshots[self.args.snapshot].num_ent] = Parameter(self.ent_embeddings.weight.data)
        new_rel_embeddings[:self.kg.snapshots[self.args.snapshot].num_rel] = Parameter(self.rel_embeddings.weight.data)

        # Update embedding attributes
        self.ent_embeddings.weight = Parameter(new_ent_embeddings)
        self.rel_embeddings.weight = Parameter(new_rel_embeddings)
        self.ent_embeddings.requires_grad = False
        self.rel_embeddings.requires_grad = False

        # Generate new LoRA embeddings
        self.lora_ent_embeddings_list_tmp, self.lora_edge_embeddings_list_tmp = self.expand_lora_embeddings()
        self.lora_ent_embeddings_list = nn.ModuleList(self.lora_ent_embeddings_list_tmp)
        self.lora_edge_embeddings_list = nn.ModuleList(self.lora_edge_embeddings_list_tmp)

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
        """
        Retrieve concatenated LoRA embeddings for entities and relations.
        """
        # Process entity embeddings
        lora_ent_embeddings = self.lora_ent_embeddings_list[0].forward(torch.arange(self.lora_ent_len).to(self.args.device))
        for lora_id in range(1, int(self.args.num_ent_layers)):
            lora_ent_embeddings = torch.cat(
                (lora_ent_embeddings, self.lora_ent_embeddings_list[lora_id].forward(torch.arange(self.lora_ent_len).to(self.args.device))),
                dim=0
            )

        # Process relation embeddings
        lora_rel_embeddings = self.lora_edge_embeddings_list[0].forward(torch.arange(self.lora_edge_len).to(self.args.device))
        for lora_id in range(1, int(self.args.num_rel_layers)):
            lora_rel_embeddings = torch.cat(
                (lora_rel_embeddings, self.lora_edge_embeddings_list[lora_id].forward(torch.arange(self.lora_edge_len).to(self.args.device))),
                dim=0
            )

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
        return self.margin_loss_func(p_score, n_score, y)

    def get_TransE_loss(self, head, relation, tail, label):
        return self.new_loss(head, relation, tail, label)

    def loss(self, head, relation, tail=None, label=None):
        loss = self.get_TransE_loss(head, relation, tail, label)
        return loss