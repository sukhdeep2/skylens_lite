#!/usr/bin/env bash

for ((j=0;j<108;j++)); do
    echo $j
    foo=$(printf "%03d" $j)
    ls /hildafs/datasets/shared_phy200017p/HSC_shape_catalog_Y1/catalog_gals_mock/4zbins | grep r$foo | wc -l
done
