# CIGA for DrugOOD IC50

공식 CIGA의 causal/spurious subgraph 분리, supervised contrastive loss, hinge loss를 유지하면서 공통 DrugOOD IC50 규약을 적용한 실행 계층이다.

## 적용 규약

- Virtual Node 없는 4-layer GIN, hidden dimension 128, dropout 0.1, sum pooling
- Adam, learning rate `1e-3`, batch size 128
- ERM pretraining 10 epochs 후 CIGA main training 최대 50 epochs
- OOD validation accuracy 기준 early stopping patience 10
- best validation checkpoint에서 test accuracy와 ROC-AUC 계산
- seeds `{1, 2, 3, 4}`의 mean/std 보고

## 단일 실행

```bash
cd /workspace/baselines/CIGA/DrugOOD

python3 train_ic50.py \
  --domain assay \
  --data-root /workspace/Graph-OOD-Lab/data/DrugOOD \
  --contrastive-weight 1.0 \
  --hinge-weight 1.0 \
  --device cuda:0
```

빠른 확인에는 `--erm-pretrain-epochs 1 --epochs 1 --num-workers 0`을 추가한다.

## 하이퍼파라미터 탐색

contrastive loss와 hinge loss 가중치를 각각 `{0.5, 1.0, 2.0, 4.0, 8.0}`에서 독립적으로 탐색하므로 seed/domain당 25개 조합이다.

```bash
python3 sweep_ic50.py \
  --domains assay \
  --seeds 1 \
  --data-root /workspace/Graph-OOD-Lab/data/DrugOOD \
  --device cuda:0
```

## 판단한 항목

- causal subgraph ratio는 기존 CIGA DrugOOD 공식 스크립트의 `0.8`을 고정 사용한다.
- assay/scaffold/size별 classifier input, contrast representation, spurious representation은 기존 공식 DrugOOD 명령을 따른다.
- ERM pretrained GIN의 shape이 일치하는 파라미터를 selector와 classifier에 전달한다. `classifier_input=feat`에서는 입력 projection shape이 달라 해당 projection만 제외된다.
- 기존 `main.py`의 `--pretrain`은 ERM pretraining이 아니라 early-stopping 시작 epoch이므로 IC50 전용 코드에서 실제 ERM 학습 단계로 분리했다.
