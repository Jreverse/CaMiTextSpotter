python projects/SWINTS/train_net.py --num-gpus 1 \
  --dist-url tcp://127.0.0.1:19208 \
  --config-file projects/SWINTS/configs/multiswints/multiswints-swin-generic.yaml \
  --eval-only MODEL.WEIGHTS output/220909/multiswints_mixtrain_full/model_0122999.pth
  

# --eval-only MODEL.WEIGHTS /dataset/s3/cv_platform/JeremyFeng/ckpt/SWINTS/multiswints/finetune-0827/model_0039999.pth