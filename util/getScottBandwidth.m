function bw = getScottBandwidth(pts)

[n,d] = size(pts);

stds = std(pts);

bw = stds.*3.5*n^(-1/3);