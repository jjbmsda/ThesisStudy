#Folder_A="/data1/xjw/vox2_train"
Folder_A="dev"  
for file_a in ${Folder_A}/*; do  
    temp_file=`basename $file_a`  
    echo $file_a
    for file_b in ${file_a}/*; do
        temp_file1=`basename $file_b`
        for file_c in ${file_b}/*; do
            temp_file2=`basename $file_c`  
		    echo $file_c
            for f in ${file_c}/*; do
            ffmpeg -i "$f" -ar 16000 "${f%.*}_t.wav"
			rm -rf "$f"
    #for file in ${temp_file}/*; do
        #temp2_file=`basename $file`
        #avconv -i "$temp2_file" "${temp2_file/%m4a/wav}"
        #echo $temp2_file
		#"${f%.*}_t.wav"
done  
done
done
done
fi
