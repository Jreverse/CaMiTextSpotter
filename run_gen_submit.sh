# first use run_eval_multi.sh gen intermediate results in tmp
task=task1
score_det=0.4
score_rec_seq=0.4
nms_iou=0.2
split=test
ckpt=model_0119999

python3 detectron2/evaluation/mlt19/mlt19_eval.py \
    --intermediate_results tmp/${ckpt}/MLT19/${split}/intermediate/ \
    --task ${task} \
    --cache_dir tmp/${ckpt}/MLT19/${split}/${task}/ \
    --score_det ${score_det} \
    --score_rec_seq ${score_rec_seq} \
    --overlap ${nms_iou} \
    --confidence_type det \
    --protocol intermediate \
    --lexicon none \
    --split ${split} \
    --seq on \
    