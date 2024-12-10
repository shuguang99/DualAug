CUDA_VISIBLE_DEVICES=0 python train_classifier.py  --logdir pascal-basicaugs/textual-inversion-0.5 \
--synthetic-dir "dualaugv2/textual-inversion-0.5/{dataset}-{seed}-{examples_per_class}" \
--dataset pascal --prompt "a photo of a {name}" \
--aug textual-inversion --guidance-scale 7.5 \
--strength 0.5 --mask 0 --inverted 0 \
--num-synthetic 10 --synthetic-probability 0.5 \
--num-trials 1 --examples-per-class 1