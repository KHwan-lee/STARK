import os
import subprocess
import sys

# 환경 변수 설정
os.environ["LANG"] = "zh_CN.UTF-8"

def run_command(command):
    """명령어를 실행하고 결과를 출력합니다."""
    print(f"실행 명령어: {command}")
    result = subprocess.run(command, shell=True, text=True)
    if result.returncode != 0:
        print(f"오류 발생: 명령어 실행 실패 (코드: {result.returncode})")
        sys.exit(result.returncode)
    return result

# 데이터 전처리
print("\n=== 데이터 전처리 ===")
# run_command("python data_to_id.py")
# run_command("python cal_features.py")
# run_command("python nodes_sort.py")

# 다양한 데이터셋에 대해 모델 학습
datasets = [
    {
        "name": "FB_CKGE",
        "ent_r": 200,
        "rel_r": 200,
        "num_ent_layers": 20,
        "num_rel_layers": 1,
        "learning_rate": "1e-1"
    },
    # {
    #     "name": "WN_CKGE",
    #     "ent_r": 200,
    #     "rel_r": 200,
    #     "num_ent_layers": 20,
    #     "num_rel_layers": 1,
    #     "learning_rate": "2e-1"
    # }
    # {
    #     "name": "FACT",
    #     "ent_r": 200,
    #     "rel_r": 200,
    #     "num_ent_layers": 20,
    #     "num_rel_layers": 1,
    #     "learning_rate": "1e-1"
    # },
    # {
    #     "name": "HYBRID",
    #     "ent_r": 200,
    #     "rel_r": 150,
    #     "num_ent_layers": 20,
    #     "num_rel_layers": 1,
    #     "learning_rate": "1e-1"
    # },
    # {
    #     "name": "ENTITY",
    #     "ent_r": 200,
    #     "rel_r": 200,
    #     "num_ent_layers": 20,
    #     "num_rel_layers": 1,
    #     "learning_rate": "2e-1"
    # },
    # {
    #     "name": "RELATION",
    #     "ent_r": 200,
    #     "rel_r": 200,
    #     "num_ent_layers": 20,
    #     "num_rel_layers": 1,
    #     "learning_rate": "3e-1"
    # },
    {
        "name": "UNIONS",
        "ent_r": 200,
        "rel_r": 200,
        "num_ent_layers": 20,
        "num_rel_layers": 1,
        "learning_rate": "1e-1",
    }
    ]

# 각 데이터셋에 대해 모델 학습 실행
for dataset in datasets:
    print(f"\n=== {dataset['name']} 데이터셋 학습 시작 ===")
    command = (
        f"python main.py -model_name LoraKGE_Layers "
        f"-ent_r {dataset['ent_r']} "
        f"-rel_r {dataset['rel_r']} "
        f"-num_ent_layers {dataset['num_ent_layers']} "
        f"-num_rel_layers {dataset['num_rel_layers']} "
        f"-gpu 0 "
        f"-dataset {dataset['name']} "
        f"-learning_rate {dataset['learning_rate']} "
        f"-using_various_ranks True "
        f"-patience 10 "
        f"-snapshot_num 1"
    )
    run_command(command)
    print(f"=== {dataset['name']} 데이터셋 학습 완료 ===")

print("\n모든 작업이 완료되었습니다.")