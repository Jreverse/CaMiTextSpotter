python demo/demo.py \
  --config-file projects/SWINTS/configs/multiswints/multiswints-swin-debug.yaml \
  --input /mnt/storage01/fengshuyang/code/swintextspotter/datasets-multi/MLT19/test/imgs/ \
  --output ./tmp/multiswints/MLT19test/ \
  --confidence-threshold 0.3 \
  --opts MODEL.WEIGHTS ./output/multiswints_finetune-0827/model_0039999.pth