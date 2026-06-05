# Run 22nm_LP with split learning, then monitor
python integrated_pipeline.py --model 22nm_LP --re-simulate no --run-tag sl --privacy-mode sl --monitor

# Run 22nm_LP with split learning and differential privacy, then monitor
python integrated_pipeline.py --model 22nm_LP --re-simulate no --run-tag both --privacy-mode both --monitor

# Run 22nm_LP differential privacy, then monitor
python integrated_pipeline.py --model 22nm_LP --re-simulate no --run-tag dp --privacy-mode dp --monitor

# Run 22nm_LP, then monitor
python integrated_pipeline.py --model 22nm_LP --re-simulate no --run-tag baseline --privacy-mode neither --monitor
