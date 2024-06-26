j=$1
MODEL=DPT
#DATASET=caltech101 # [eurosat caltech101 oxford_flowers food101 fgvc_aircraft]
#DATADIR=caltech101 #[EuroSAT Caltech101 Flowers102 Food101 FGVC/fgvc-aircraft-2013b]
# DATASET=caltech101
# DATADIR=Caltech101
#DATASET=eurosat
#DATADIR=EuroSAT
# DATASET=stanford_cars
# DATADIR=StanfordCars
# DATASET=oxford_flowers
# DATADIR=Flowers102
# DATASET=dtd
# DATADIR=DTD
# DATASET=food101
# DATADIR=Food101
# DATASET=oxford_pets
# DATADIR=OxfordPets
# DATASET=imagenet
# DATADIR=ImageNet1K
# DATASET=sun397
# DATADIR=Sun397
# DATASET=ucf101
# DATADIR=UCF101

# CPN is the length of CAVPT
# BOTTOMLIMIT is the layers CAVPT inserted into. e.g. 8 means 8-12 layers are equipped with CAVPT. 12 means 12 layers are equipped with CAVPT.1 means every layer are equipped with CAVPT.
# C is our general knowledge guidence epoch.
# ALPHA is loss balancing parameter.
for DATASET in caltech101 dtd oxford_pets stanford_cars ucf101 food101 sun397 fgvc_aircraft eurosat oxford_flowers imagenet
do
  for N_SHOTS in 1 2 4 8 16
  do
    python train.py --root ../Tip-Adapter/data/ --seed $j \
    --trainer $MODEL \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --output-dir ./output/${DATASET} \
    --config-file ./configs/trainers/VPT/vit_b16.yaml \
    TRAINER.COOP.N_CTX 16 \
    TRAINER.COOP.CSC False \
    TRAINER.COOP.CLASS_TOKEN_POSITION end \
    DATASET.NUM_SHOTS ${N_SHOTS} \
    TRAINER.VPT.N_CTX 10 \
    TRAINER.TOPDOWN_SECOVPT.BOTTOMLIMIT 12 \
    TRAINER.SELECTED_COVPT.CPN 10 \
    OPTIM.LR 0.01 \
    OPTIM.MAX_EPOCH 60 \
    PRETRAIN.C 30 \
    TRAINER.ALPHA 0.3
  done
done

