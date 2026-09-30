DATASET=${1}
LAST=${2:-""}
if [[ -n "$LAST" ]]; then
    LAST="_$LAST"
fi
TYPE=${3:-"tiger"}

export CUDA_VISIBLE_DEVICES=0,1,2,3
tokenizer_lr=1e-3
model_bsz=1024
uv run accelerate launch --num_processes=4 train_rqvae.py\
    tokenizer.data_text_files="./data/$DATASET$LAST/pkl/item2title.pkl"\
    tokenizer.interaction_files="./data/$DATASET$LAST/pkl/user2item.pkl"\
    tokenizer.text_encoder_model="sentence-t5-base"\
    type="$TYPE"\
    output_dir="${DATASET}${LAST}_$TYPE"\
    dataset="$DATASET$LAST" \
    tokenizer.learning_rate=$tokenizer_lr \
    tokenizer.batch_size=$model_bsz \
    seed=1000
    
    
    
