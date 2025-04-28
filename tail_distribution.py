import os
import csv

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path
import seaborn as sns

# def merge_triples(dataset_name):
#     """
#     각 데이터셋의 0~4 스냅샷의 트리플을 합치고 중복을 제거하여 all_triples.txt로 저장
    
#     Args:
#         dataset_name: 데이터셋 이름
#     """
#     base_path = f'data/{dataset_name}'
#     all_triples = set()
    
#     # 0~4 스냅샷의 train, valid, test 파일 읽기
#     for snapshot in range(5):
#         snapshot_path = os.path.join(base_path, str(snapshot))
#         for file_name in ['train.txt', 'valid.txt', 'test.txt']:
#             file_path = os.path.join(snapshot_path, file_name)
#             if os.path.exists(file_path):
#                 with open(file_path, 'r') as f:
#                     for line in f:
#                         all_triples.add(line.strip())
    
#     # 4번 스냅샷에 all_triples.txt 저장
#     output_path = os.path.join(base_path, '4', 'all_triples.txt')
#     with open(output_path, 'w') as f:
#         for triple in sorted(all_triples):
#             f.write(triple + '\n')
    
#     print(f"{dataset_name}의 모든 트리플이 {output_path}에 저장되었습니다.")


# 모든 데이터셋의 최종 임베딩 비교


# 6개 데이터셋 정의, snapshot 4
# datasets = ['RELATION', 'ENTITY', 'HYBRID', 'FB_CKGE', 'FACT']
# snapshot = 4

# # 색상 팔레트 준비 (HUSL)
# colors = dict(zip(datasets, sns.color_palette('husl', n_colors=len(datasets))))

# # 엔티티 임베딩 로드 및 레이블 생성
# all_embeddings = []
# all_labels = []
# for ds in datasets:
#     emb_path = Path('embeddings') / ds / str(snapshot) / 'entity_embeddings.npy'
#     if not emb_path.exists():
#         print(f"[경고] {emb_path} 파일을 찾을 수 없습니다.")
#         continue
#     emb = np.load(emb_path)
#     all_embeddings.append(emb)
#     all_labels.extend([ds] * emb.shape[0])

# # 결합
# concat_emb = np.vstack(all_embeddings)

# # PCA 적용
# pca = PCA(n_components=2, random_state=42)
# emb_2d = pca.fit_transform(concat_emb)

# # 시각화
# plt.figure(figsize=(14, 10))
# start = 0
# for ds in datasets:
#     # 각 데이터셋의 개체 수
#     emb_path = Path('embeddings') / ds / str(snapshot) / 'entity_embeddings.npy'
#     if not emb_path.exists():
#         continue
#     num = np.load(emb_path).shape[0]
#     end = start + num
#     pts = emb_2d[start:end]
#     plt.scatter(pts[:, 0], pts[:, 1],
#                 color=colors[ds], alpha=0.6, s=5, label=ds)
#     start = end

# plt.title(f'Combined PCA of Snapshot {snapshot} Entity Embeddings', fontsize=16)
# plt.xlabel('PC 1', fontsize=14)
# plt.ylabel('PC 2', fontsize=14)
# plt.legend(title='Dataset', fontsize=12, title_fontsize=13)
# plt.grid(True)
# plt.tight_layout()
# plt.show()


## 각 데이터셋의 스냅샷별 임베딩 비교

def visualize_pca_by_snapshot(embeddings_root, datasets, snapshots):
    """
    각 데이터셋별로:
    - 스냅샷 k까지 누적된 전체 엔티티 임베딩을 (최대 sample_limit)만큼 샘플링
    - PCA로 2D로 축소 후, 스냅샷별로 색을 달리해 한 그래프에 표시
    """
    for dataset in datasets:
        plt.figure(figsize=(14, 10))
        if  dataset == 'UNIONS':
            emb_path = Path(embeddings_root) / dataset / '0' / 'entity_embeddings.npy'
        else:
            emb_path = Path(embeddings_root) / dataset / '0' / 'entity_embeddings.npy'

        emb = np.load(emb_path)
        pca = PCA(n_components=2, random_state=42).fit(emb)

        for idx, snap in enumerate(snapshots):
            emb_path = Path(embeddings_root) / dataset / str(snap) / 'entity_embeddings.npy'
            if not emb_path.exists():
                print(f"[경고] {emb_path} 없음, 건너뜁니다.")
                continue

            emb = np.load(emb_path)
            emb2d = pca.transform(emb)

            plt.scatter(
                emb2d[:, 0], emb2d[:, 1],
                color=colors[idx],
                alpha=1 - 0.2 * snap,
                s=5,
                label=f'Snap {snap}'
            )

        plt.title(f'{dataset} Cumulative Entity Embeddings PCA', fontsize=16)
        plt.xlabel('PC1', fontsize=14)
        plt.ylabel('PC2', fontsize=14)
        plt.legend(fontsize=12, title='Snapshot', title_fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# 6개 데이터셋과 5개 스냅샷 정의
datasets = ['UNIONS', 'FB_CKGE',  'FACT', 'RELATION', 'ENTITY', 'HYBRID']
snapshots = [0, 1, 2, 3, 4]

# 스냅샷마다 구분할 색상(Tab10 컬러맵)
colors = sns.color_palette('husl', n_colors=len(snapshots))


# 사용 예시
if __name__ == '__main__':
    visualize_pca_by_snapshot('embeddings', datasets, snapshots)
    #pass


# # 각 데이터셋에 대해 실행
# for dataset in datasets:
#     merge_triples(dataset)