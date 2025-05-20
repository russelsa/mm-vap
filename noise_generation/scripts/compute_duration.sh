R=0
for a in /data/ssd4/lrs3_babble/val/*.wav
do
  T="$(soxi -D $a 2>-)"
  echo $T
  [[ "$T" != "" ]] && R="$R + $T"
done
echo $R | bc