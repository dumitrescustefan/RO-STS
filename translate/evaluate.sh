for checkpoint in $2/*.pt; do
    echo "$checkpoint"
    output="$3/pred_$(echo "$(basename $checkpoint)" | cut -f 1 -d '.')"
    onmt_translate -config $1 --model $checkpoint --output "$output.sp"
    spm_decode -model=$6 < "$output.sp" > "$output.txt"
    sacrebleu "$output.txt" < $4 > "$5/$(echo "$(basename $checkpoint)" | cut -f 1 -d '.').txt"
done