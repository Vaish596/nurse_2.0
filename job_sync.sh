#comment everything and run deface command to deface any video
sh install.sh

SESSION_ID=$1
VIDEO_FILE=$2
# deface -t 0.6 -o /netscratch/shirbhata/nurse_2.0/10383_all_cams_anonymized.mp4 /ds/videos/nurse_2.0/videos_sync/1080p/session_1/10383_all_cams.mp4
# python cluster_int_sensor_ani_new.py

python cluster_int_sensor_ani_new.py \
  --session_id "$SESSION_ID" \
  --video_file "$VIDEO_FILE"
# echo 'Done generating'
