#!/bin/bash

# This script will initialize a new project

echo 'Creating New Project'
echo ''

mkdir models

module load python

echo 'Creating Virtual Environment'
echo ''
python -m venv env

source env/bin/activate

pip install --upgrade pip

pip3 install  torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 numpy -f https://download.pytorch.org/whl/torch_stable.html matplotlib scikit-learn --extra-index-url https://download.pytorch.org/whl/cu113

echo 'Creating Requirements File'
echo ''

pip freeze > requirements.txt

echo 'Creating Job Script'
echo '
#!/bin/bash

#SBATCH -p
#SBATCH gpu
#SBATCH --gpus=a100:1
#SBATCH --account=eee4773
#SBATCH --qos=eee4773
#SBATCH --mem-per-gpu=10gb
#SBATCH --time=02:00:00

module load python
module load cuda/11.1.0

# activate python environment
source env/bin/activate

nvidia-smi

python SimpleCNN.py $TIME'>job.slurm
echo 'Creating Launch Script'
echo ''

echo "#!/bin/bash" > launch.sh
echo "TIME=\`date +\%s\`" >> launch.sh
echo "mkdir models/\$TIME" >> launch.sh
echo "sbatch --output=models/\$TIME/log.out --export=TIME=\$TIME job.slurm" >> launch.sh

echo "Initialization Complete"
