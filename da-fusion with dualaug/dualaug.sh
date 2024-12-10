CUDA_VISIBLE_DEVICES=1 python train_classifier_dualaug.py  --logdir pascal-dualaugs/textual-inversion-0.5 \
--synthetic-dir "dualaug/textual-inversion-0.5/{dataset}-{seed}-{examples_per_class}" \
--dataset pascal --prompt "a photo of a {name}" \
--aug textual-inversion --guidance-scale 7.5 \
--strength 0.5 --mask 0 --inverted 0 \
--num-synthetic 10 --synthetic-probability 0.5 \
--num-trials 1 --examples-per-class 1