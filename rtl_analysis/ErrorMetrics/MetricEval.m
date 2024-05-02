ApproxCircuitOutputSpace = readmatrix("output.csv");

for i = 1:size(ApproxCircuitOutputSpace)


f = figure('Position',[10,10,600,400])
hold on;
histogram(ApproxCircuitOutputSpace)