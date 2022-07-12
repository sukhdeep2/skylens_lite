#!/usr/bin/env bash

datetime="`date +%d%H%M%S`"
JobName="andy-mpi-"$datetime
foo="#!/usr/bin/env bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 20
#SBATCH --time 04:00:00
#SBATCH --job-name $JobName
#SBATCH --output andy-mpi-out-$datetime
#SBATCH --error andy-mpi-err-$datetime
#SBATCH --mem=80G
$(export)
$(export -f)
cd $PWD
$@
"

cd $PWD &&
echo "${foo@E}" | sbatch
