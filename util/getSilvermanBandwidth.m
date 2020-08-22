function bw = getSilvermanBandwidth(pts)

[n,d] = size(pts);

stds = std(pts);

bw = stds.*((4/((d+2)*n))^(1/(d+4)));