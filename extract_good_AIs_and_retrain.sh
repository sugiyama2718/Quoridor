echo $1
python extract_good_AIs.py $1
python main.py retrain 2> error_log.txt
