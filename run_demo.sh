python demo/demo.py \
  --config-file projects/SWINTS/configs/SWINTS/SWINTS-swin-debug.yaml \
  --input /mnt/storage01/fengshuyang/code/swintextspotter/datasets/debug/test_images/ \
  --output ./tmp/multiswints_finetune-0827/ \
  --confidence-threshold 0.4 \
  --opts MODEL.WEIGHTS ./output/multiswints_finetune-0827/model_0039999.pth