python3 run_experiment.py --experiment_name FREEZE_EMB --config configs/SberZvuk_discrete_settings.yaml --framestack 20 --model_parametrs 1000 1000 16 32 20 64 0 0 --freeze_emb 1 --use_als 0
python3 run_experiment.py --experiment_name FREEZE_EMB --config configs/SberZvuk_discrete_settings.yaml --framestack 20 --model_parametrs 1000 1000 16 32 20 64 1 64 --freeze_emb 1 --use_als 0

python3 run_experiment.py --experiment_name FREEZE_EMB --config configs/SberZvuk_discrete_settings.yaml --framestack 20 --model_parametrs 1000 1000 8 32 20 64 0 0 --freeze_emb 1 --use_als 0
python3 run_experiment.py --experiment_name FREEZE_EMB --config configs/SberZvuk_discrete_settings.yaml --framestack 20 --model_parametrs 1000 1000 8 32 20 64 1 64 --freeze_emb 1 --use_als 0

python3 run_experiment.py --experiment_name FREEZE_EMB --config configs/SberZvuk_discrete_settings.yaml --framestack 20 --model_parametrs 1000 1000 16 64 20 128 0 0 --freeze_emb 1 --use_als 0
python3 run_experiment.py --experiment_name FREEZE_EMB --config configs/SberZvuk_discrete_settings.yaml --framestack 20 --model_parametrs 1000 1000 16 64 20 128 1 128 --freeze_emb 1 --use_als 0

python3 run_experiment.py --experiment_name FREEZE_EMB --config configs/SberZvuk_discrete_settings.yaml --framestack 20 --model_parametrs 1000 1000 16 128 20 256 0 0 --freeze_emb 1 --use_als 0
python3 run_experiment.py --experiment_name FREEZE_EMB --config configs/SberZvuk_discrete_settings.yaml --framestack 20 --model_parametrs 1000 1000 16 128 20 256 1 256 --freeze_emb 1 --use_als 0