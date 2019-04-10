export BASE_DIR=$HOME/Humanware_v1_1551895483
export TRAIN_SET=$BASE_DIR/train
export VALID_SET=$BASE_DIR/valid
export TEST_SET=$DATA_DIR
# export TEST_SET=$BASE_DIR/test
TEAM_NAME=b3phut_baseline


mkdir -p $HOME/$TEAM_NAME/code/maskrcnn-mila/datasets/avenue
mkdir -p $HOME/$TEAM_NAME/code/maskrcnn-mila/datasets/avenue/annotations

# SymLink datassets
ln -sfn $TRAIN_SET $HOME/$TEAM_NAME/code/maskrcnn-mila/datasets/avenue/train
ln -sfn $VALID_SET $HOME/$TEAM_NAME/code/maskrcnn-mila/datasets/avenue/validation
ln -sfn $TEST_SET $HOME/$TEAM_NAME/code/maskrcnn-mila/datasets/avenue/test

# Symlink annotations
ln -sfn $TRAIN_SET/instances_train.json $HOME/$TEAM_NAME/code/maskrcnn-mila/datasets/avenue/annotations/instances_train_avenue.json
# ln -sfn ~/avenue_train/instances_avenue.json ~/$HOME/$TEAM_NAME/code/maskrcnn-mila/datasets/avenue/annotations/instances_train_avenue.json

ln -sfn $VALID_SET/instances_valid.json $HOME/$TEAM_NAME/code/maskrcnn-mila/datasets/avenue/annotations/instances_valid_avenue.json
# ln -sfn ~/avenue_train/instances_avenue.json ~/$HOME/$TEAM_NAME/code/maskrcnn-mila/datasets/avenue/annotations/instances_valid_avenue.json

ln -sfn $TEST_SET/instances_test.json $HOME/$TEAM_NAME/code/maskrcnn-mila/datasets/avenue/annotations/instances_test_avenue.json
