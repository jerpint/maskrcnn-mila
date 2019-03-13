export BASE_DIR=/network/home/$USER/Humanware_v1_1551895483
export TRAIN_SET=$BASE_DIR/train
export VALID_SET=$BASE_DIR/valid
export TEST_SET=$BASE_DIR/test

mkdir -p ~/maskrcnn-mila/datasets/avenue

mkdir -p ~/maskrcnn-mila/datasets/avenue/annotations

# SymLink datassets
ln -sfn $TRAIN_SET ~/maskrcnn-mila/datasets/avenue/train
ln -sfn $VALID_SET ~/maskrcnn-mila/datasets/avenue/validation
ln -sfn $TEST_SET ~/maskrcnn-mila/datasets/avenue/test

# Symlink annotations
ln -sfn $TRAIN_SET/instances_train.json ~/maskrcnn-mila/datasets/avenue/annotations/instances_train_avenue.json
# ln -sfn ~/avenue_train/instances_avenue.json ~/maskrcnn-mila/datasets/avenue/annotations/instances_train_avenue.json

ln -sfn $VALID_SET/instances_valid.json ~/maskrcnn-mila/datasets/avenue/annotations/instances_valid_avenue.json
# ln -sfn ~/avenue_train/instances_avenue.json ~/maskrcnn-mila/datasets/avenue/annotations/instances_valid_avenue.json

ln -sfn $TEST_SET/instances_test.json ~/maskrcnn-mila/datasets/avenue/annotations/instances_test_avenue.json
