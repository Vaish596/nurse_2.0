you can run the program with command: ./launch_job.sh <session_id> <input_file)name>

eg: ./launch_job.sh 383 /netscratch/shirbhata/nurse_2.0/383_all_cams_anonymized.mp4

To blur out faces in video files, run inside job.sh command: deface -t 0.6 -o <output_path> <input_path>
eg: deface -t 0.6 -o /netscratch/shirbhata/nurse_2.0/10383_all_cams_anonymized.mp4 /ds/videos/nurse_2.0/videos_sync/1080p/session_1/10383_all_cams.mp4
