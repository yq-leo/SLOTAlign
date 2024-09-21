python SLOTAlign.py --dataset=Douban --runs=10 --record --robust --exp_name=attr_noise
for attr_noise in $(seq 0.1 0.1 0.9); do
  python SLOTAlign.py --dataset=Douban --runs=10 --attr_noise_rate=$attr_noise --strong_noise --record --robust --exp_name=attr_noise
done
