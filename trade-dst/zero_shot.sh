AUG='KR18973_lpt_nfs3_6030.json'
EXCEPT='hotel'
DO_ALL=true
domains=( 'hotel' 'restaurant' 'attraction' 'train' 'taxi' )
# augs=( 'KR18973_lpt_nfs3_4633.json' 'KR18973_lpt_nfs3_4666.json' 'KR18973_lpt_nfs3_4667.json' 'KR18973_lpt_nfs3_4669.json' 'KR18973_lpt_nfs3_4670.json' )
#augs=( '4765' '4766' '4767' '4768' '4769' )
#augs=( '5029' '5030' '5031' '5032' '5033' )

if [ $DO_ALL == true ]
then

for (( i=0; i<${#domains[@]}; i++ ));
    do
       nsml run -d lpt_nfs3 -g 1 -e myTrain.py -a "-dec=TRADE -bsz=32 -dr=0.2 -lr=0.001 -le=1 -clip=1 -td=/nsml_nfs_output/sungdong/neuralwoz/nwoz_${domains[i]}_0.json --dataset_path=/nsml_nfs_output/sungdong/neuralwoz/data -ed ${domains[i]} --output_path=/nsml_nfs_output/sungdong/neuralwoz/${domains[i]}_concat"  -m "zeroshot with augmented for ${domains[i]}; clip 1; concat" --nfs-output
    done
else

if [ $EXCEPT != 'None' ]
then
  o1="-ed=$EXCEPT"
else
  o1=""
fi

if [ $AUG != 'None' ]
then
  o2="-aug=$AUG"
else
  o2=""
fi

cmd="nsml run -d lpt_nfs3 -g 1 -e myTrain.py -m '$EXCEPT' -a \"-dec=TRADE -bsz=32 -dr=0.2 -lr=0.001 -le=1 $o1 $o2\" --nfs-output"

echo $cmd

fi