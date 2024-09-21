python SLOTAlign.py --dataset=Douban --runs=10 --record --robust --exp_name=attr_noise
for %%N in (0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9) do (
    python SLOTAlign.py --dataset=Douban --runs=10 --attr_noise_rate=%%N --strong_noise --record --robust --exp_name=attr_noise
)