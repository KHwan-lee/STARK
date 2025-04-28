import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import random
import os
from pathlib import Path
from nltk.corpus import wordnet as wn
import csv


mid2name = {}
with open("fb_mapping.tsv", encoding="utf‑8") as f:
    for mid, name1, name2 in csv.reader(f, delimiter="\t"):
        mid2name[mid] = name2

# print(mid2name["/m/02mjmr"])   # → Brad Pitt


def id2label(raw_id, freebase_map, pos_default='n'):

    if raw_id.startswith('/m/'):                 # Freebase
        return freebase_map.get(raw_id, raw_id)  # 없으면 그대로 반환
    else:                                        # WordNet
        print("WN-CKGE")
        try:
            offset = int(raw_id.lstrip('0'))  # 선행 0 제거
        except ValueError:
            return raw_id                     # 숫자가 아니면 그대로 반환

        for pos in ['n', 'v', 'a', 's', 'r']:
            syn = wn.synset_from_pos_and_offset(pos, offset)
            if syn is not None:
                return f"{syn.name()} — {syn.definition()}"

    # 아무 품사에서도 못 찾았을 때
    return f"[unknown WordNet offset {raw_id}]"

# 예시
# print(id2label('/m/02mjmr', mid2name))   # Brad Pitt
# print(id2label('10160913', mid2name))    # dog.n.01




def analyze_and_visualize_triples(dataset_name, snapshot=4):
    """
    데이터셋의 트리플을 분석하고 시각화하는 함수
    
    Args:
        dataset_name: 데이터셋 이름 (예: 'WN_CKGE', 'FB_CKGE' 등)
        snapshot: 스냅샷 번호 (0부터 시작)
    """
    base_path = f'data/{dataset_name}/{snapshot}/'
    
    # 1. 데이터 로딩
    def load_triples(file_path):
        triples = []
        with open(file_path, 'r') as f:
            for line in f:
                h, r, t = line.strip().split('\t')
                triples.append((h, r, t))
        return triples

    triples = load_triples(base_path + 'all_triples.txt')
    #test_triples = load_triples(base_path + 'test.txt')
    
    # 2. 관계 패턴 분석
    head_rel_to_tail = defaultdict(set)
    tail_rail_to_head = defaultdict(set)
    
    # 학습 데이터로 관계 패턴 구축
    for h, r, t in triples:
        head_rel_to_tail[(h, r)].add(t)
        tail_rail_to_head[(t, r)].add((h))
    
    # 테스트 트리플 분류
    relation_patterns = {
        '1-1': [], # 1대1 관계
        'N-1': [], # 다대1 관계
        '1-N': [], # 1대다 관계
        'N-N': []  # 다대다 관계
    }
    
    for h, r, t in triples:
        # head에서 나가는 관계 수 (head가 주어진 관계 r로 연결된 tail의 수)
        head_out = len(head_rel_to_tail[(h, r)])
        # tail로 들어오는 관계 수 (tail이 주어진 관계 r로 연결된 head의 수)
        tail_in = len(tail_rail_to_head[(t, r)])
        
        # 1-1 관계: head가 하나의 tail과만 연결, tail도 하나의 head와만 연결
        if head_out == 1 and tail_in == 1:
            relation_patterns['1-1'].append((h, r, t))
        # 1-N 관계: 여러 head가 하나의 tail과 연결
        elif head_out > 1 and tail_in == 1:
            relation_patterns['1-N'].append((h, r, t))
        # N-1 관계: 하나의 tail가 여러 head과 연결
        elif head_out == 1 and tail_in > 1:
            relation_patterns['N-1'].append((h, r, t))
        # N-N 관계: 여러 head가 여러 tail과 연결
        else:
            relation_patterns['N-N'].append((h, r, t))
    
    # 3. 결과 출력
    print(f"\n=== {dataset_name} 데이터셋 (스냅샷 {snapshot}) 분석 결과 ===")
    for pattern, triples in relation_patterns.items():
        print(f"{pattern} 관계: {len(triples)} 트리플")
    
    return relation_patterns

def visualize_embeddings_by_pattern(embeddings_path, relation_embeddings_path, relation_patterns, pattern_type_num):
    """
    선택된 패턴 유형에 대해 랜덤하게 (head/tail, relation) 쌍을 선택하여 임베딩 시각화
    TransE의 정답 지점도 함께 표시
    
    Args:
        embeddings_path: 엔티티 임베딩 파일 경로
        relation_embeddings_path: 관계 임베딩 파일 경로
        relation_patterns: 관계 패턴 분류 결과
        pattern_type_num: 패턴 유형 번호 (1: 1-1, 2: 1-N, 3: N-1, 4: N-N)
    """
    pattern_map = {1: '1-1', 2: '1-N', 3: 'N-1', 4: 'N-N'}
    
    if pattern_type_num not in pattern_map:
        print("잘못된 패턴 타입입니다. 1, 2, 3, 4 중 하나를 선택하세요.")
        return
        
    pattern_type = pattern_map[pattern_type_num]
    triples = relation_patterns[pattern_type]
    print(len(triples))
    
    if not triples:
        print(f"{pattern_type} 패턴에 해당하는 트리플이 없습니다.")
        return
    
    # 랜덤하게 트리플 선택
    selected_triple = random.choice(triples)
    head, relation, tail = selected_triple
    

    entity2id = {}
    p = Path(embeddings_path)

    ###PCA 차원축소
    emb = np.load(embeddings_path)
    pca = PCA(n_components=2, random_state=42)
    emb_2d = pca.fit_transform(emb)
    print(emb_2d.shape)
    

    plt.figure(figsize=(10, 7))
    title_text = f' Entity Embeddings Distribution for {p.parts[1]}'
    plt.title(f'{title_text}')
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=0.5, c='gray', s=25)
    plt.show()
    ## PCA 차원축소 끝


    base_path = f'data/{p.parts[1]}/{p.parts[2]}/'
    entity2id_path = os.path.join(base_path, "entity2id.txt")

    with open(entity2id_path, "r", encoding="utf-8") as rf:
        lines = list(rf.readlines())
        for line in lines:
            line_split = line.strip().split()
            ent, ent_id = line_split[0], line_split[1]
            entity2id[ent] = int(ent_id)

    relation2id = {}
    relation2id_path = os.path.join(base_path, "relation2id.txt")
    with open(relation2id_path, "r", encoding="utf-8") as rf:
        lines = list(rf.readlines())
        for line in lines:
            line_split = line.strip().split()
            rel, rel_id = line_split[0], line_split[1]
            relation2id[rel] = int(rel_id)
    # 엔티티와 관계 임베딩 로드

    entity_embeddings = np.load(embeddings_path)
    relation_embeddings = np.load(relation_embeddings_path)
    
    # N-1 패턴의 경우 tail과 relation 기준으로 head 임베딩 시각화
    if pattern_type_num == 3:
        target_entities = [entity2id[head]]
        other_entities = []
        
        for h, r, t in triples:
            if t == tail and r == relation and h != head:
                other_entities.append(entity2id[h])

        title_text = f'Tail Entity Embeddings Distribution\nfor (t={id2label(tail, mid2name)}, r={relation})'
        #target_label = f'Embedding for (h={id2label(head, mid2name)})'
        
        # TransE 정답 지점 계산 (tail - relation)
        answer_point = entity_embeddings[int(entity2id[tail])] - relation_embeddings[int(relation2id[relation])]
        #answer_label = 't - r'
    
    # 다른 패턴들의 경우 head와 relation 기준으로 tail 임베딩 시각화
    else:

        target_entities = [entity2id[tail]]
        other_entities = []
        
        for h, r, t in triples:
            if h == head and r == relation and t != tail:
                other_entities.append(entity2id[t])

        title_text = f'Tail Entity Embeddings Distribution\nfor (h={id2label(head, mid2name)}\n r={relation})'
        #target_label = f'Embedding for (t={mid2name[tail]})'

        answer_point = entity_embeddings[int(entity2id[head])] + relation_embeddings[int(relation2id[relation])]
        #answer_label = 'h + r'

    
    if not target_entities:
        print("선택된 엔티티 쌍에 대한 대상 엔티티가 없습니다.")
        return
        
    # t-SNE로 차원 축소
    target_embeddings = entity_embeddings[target_entities]
    
    # if other_entities:
    #     sample_size = min(100, len(other_entities))
    #     other_entities = random.sample(other_entities, sample_size)
    other_embeddings = entity_embeddings[other_entities]
    
    # 랜덤 임베딩 추가(비교군)
    all_entity_ids = set(range(entity_embeddings.shape[0]))
    excluded_entities = set([entity2id[tail]] + other_entities)
    candidate_entities = list(all_entity_ids - excluded_entities)
    random_entities = random.sample(candidate_entities, max(min(len(other_entities) * 2, len(candidate_entities)),100))
    random_embeddings = entity_embeddings[random_entities]
    # 정답 지점을 포함한 모든 임베딩 준비
    all_embeddings = np.vstack([target_embeddings, other_embeddings, answer_point, random_embeddings])


    # TSNE 차원축소
    embeddings_1= TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42).fit_transform(all_embeddings)
    
    #PCA 차원축소
    pca = PCA(n_components=2, random_state=42)
    embeddings_2 = pca.fit_transform(all_embeddings)

    embeddings = [embeddings_1, embeddings_2]

    # 시각화
    for embeddings_2d in embeddings:
        plt.figure(figsize=(10, 7))
        

        idx = 1

        # (h,r) 그룹의 다른 tail (투명 초록)
        if len(other_embeddings) > 0:
            if(pattern_type_num == 3):
                plt.scatter(
                    embeddings_2d[idx:idx+len(other_embeddings), 0],
                    embeddings_2d[idx:idx+len(other_embeddings), 1],
                    c='green', alpha=0.7, s=30, label='Other tails (same t,r)'
                )
            else:
                plt.scatter(
                    embeddings_2d[idx:idx+len(other_embeddings), 0],
                    embeddings_2d[idx:idx+len(other_embeddings), 1],
                    c='green', alpha=0.7, s=30, label='Other tails (same h,r)'
                )
            idx += len(other_embeddings)

        # 랜덤 엔티티 (회색)
        plt.scatter(
            embeddings_2d[idx:idx+len(random_embeddings), 0],
            embeddings_2d[idx:idx+len(random_embeddings), 1],
            c='gray', alpha=0.2, s=20, label='Random entities'
        )
        idx += len(random_embeddings)

        # 이상적인 정답 지점 (파란 별)
        if(pattern_type_num == 3):
            plt.scatter(
                embeddings_2d[-1, 0], embeddings_2d[-1, 1],
                c='blue', marker='*', s=300, label='Ideal (t - r)'
            )
        else:
            plt.scatter(
                embeddings_2d[-1, 0], embeddings_2d[-1, 1],
                c='blue', marker='*', s=300, label='Ideal (h + r)'
            )
        
        idx = 0  # 시작 인덱스

        # 실제 tail (빨간 별)
        if(pattern_type_num == 3):
            plt.scatter(
                embeddings_2d[idx, 0], embeddings_2d[idx, 1],
                c='red', s=200, marker='*', label=f'Actual Head: {id2label(head, mid2name)}'
            )
        else:
            plt.scatter(
                embeddings_2d[idx, 0], embeddings_2d[idx, 1],
                c='red', s=200, marker='*', label=f'Actual Tail: {id2label(tail, mid2name)}'
            )

        plt.title(f'{title_text}\n({pattern_type} relation)')
        plt.legend()
        
        # 통계 정보 출력
        print(f"\n=== 선택된 트리플 정보 ===")
        print(f"패턴 타입: {pattern_type}")
        if pattern_type_num == 3:
            print(f"Tail: {tail}")
            print(f"Relation: {relation}")
            print(f"Target head 엔티티 수: {len(target_entities)}")
        else:
            print(f"Head: {head}")
            print(f"Relation: {relation}")
            print(f"Target tail 엔티티 수: {len(target_entities)}")
        print(f"Other 엔티티 수: {len(other_entities)}")
        
        plt.show()

#사용 예시
#datasets = ['FB_CKGE', 'FACT', 'HYBRID', 'ENTITY', 'RELATION', 'WN_CKGE']
datasets = ['WN_CKGE']


for dataset in datasets:
    try:
        patterns = analyze_and_visualize_triples(dataset)
        
        # pattern_type_num = int(input(f"\n{dataset} 데이터셋에 대해 시각화할 패턴 타입을 선택하세요 (1:1-1, 2:1-N, 3:N-1, 4:N-N): "))
        
        # 엔티티와 관계 임베딩 파일 경로
        embeddings_path = f'embeddings/{dataset}/4/entity_embeddings.npy'
        relation_embeddings_path = f'embeddings/{dataset}/4/relation_embeddings.npy'
        
        visualize_embeddings_by_pattern(embeddings_path, relation_embeddings_path, patterns, pattern_type_num=2)
        
    except Exception as e:
        print(f"{dataset} 처리 중 오류 발생: {str(e)}")


# 예시
#patterns = analyze_and_visualize_triples('WN_CKGE')
#visualize_tail_embeddings('embeddings/WN_CKGE_entity_embeddings.npy', 'head_id', 'relation_id', patterns, '1-N')