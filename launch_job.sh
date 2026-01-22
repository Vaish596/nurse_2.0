#512
srun \
  --cpus-per-task=9 \
  --job-name='nurse_385' \
  --gpus=0 \
  --mem="40GB" \
  --partition=batch,H100-RP,A100-80GB,H200,A100-40GB,RTXA6000,V100-32GB \
  --ntasks=1 \
  --container-mounts="/ds":"/ds","/netscratch":"/netscratch","`pwd`":"`pwd`"  \
  --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_23.04-py3.sqsh \
  --container-workdir="`pwd`" \
  --time=01-00:00 \
  bash job.sh 