#how to run
# 以 Logistic Regression 为例（保证已用 prepare_dataset.py 生成 imdb_clean.csv） :contentReference[oaicite:11]{index=11}
python experimental_pipeline01.py --csv "E:\sml\imdb_clean.csv" --model lr --outer_k 10 --inner_k 3

# BiLSTM
python experimental_pipeline01.py --csv "E:\sml\imdb_clean.csv" --model bilstm --outer_k 10 --inner_k 3

# ELECTRA（首次会下载权重）
python experimental_pipeline01.py --csv "E:\sml\imdb_clean.csv" --model electra --outer_k 10 --inner_k 3
