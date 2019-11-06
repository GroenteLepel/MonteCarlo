bw = .0025
set boxwidth bw
bin(x,width) = width*floor(x/width) + bw/2.0
set xrange [0:*]
set yrange [0:*]
plot "rcarry_generate.txt" using (bin($1,bw)) smooth freq with boxes
