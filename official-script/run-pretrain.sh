DATASET=${1}
LAST=${2:-""}
if [[ -n "$LAST" ]]; then
    LAST="_$LAST"
fi
TYPE=${3:-"tiger"}

export CUDA_VISIBLE_DEVICES=6,7
model_lr=2e-3
model_decay=0.2
model_bsz=1024
num_epochs=200
uv run accelerate launch --num_processes=2 train_with_generative.py \
    model.data_interaction_files="./data/$DATASET$LAST/pkl/user2item.pkl" \
    model.data_text_files="./data/$DATASET$LAST/pkl/item2title.pkl" \
    model.learning_rate=$model_lr \
    model.weight_decay=$model_decay \
    model.batch_size=$model_bsz \
    model.num_epochs=$num_epochs \
    model.early_stop_upper_steps=20\
    model.evaluation_epoch=1\
    output_dir="${DATASET}${LAST}_$TYPE"\
    generative="$TYPE"
