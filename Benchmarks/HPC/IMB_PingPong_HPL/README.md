# ssh to Megware, ssh frontend
srun -N 1 --partition=CPU_6140  --nodelist=ibp05,ibp06 --pty bash


./HPL_run.sh&

srun -N 1 --partition=CPU_6140  --nodelist=ibp06 --pty bash


./HPL_run.sh&

# On the first srun (double nodelist), launch:
./IMB_run.sh


