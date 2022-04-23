#! /bin/bash
export PATH=$PATH:$JAVA_HOME/bin/
dir="$(dirname $1)"
base="$(basename $1)"
mkdir $dir/tmp_process
mkdir $dir/tmp_process/output
python3 split_json.py --input_json $1
echo "phrase extraction"
for fn in `ls $dir/tmp_process`;
do
echo "process $fn"
java -Xmx8G -jar spice/spice-1.0.jar $dir/tmp_process/$fn -out $dir/tmp_process/output/$fn -threads 20 -detailed --silent
done
python3 merge_json.py --input_data $1
