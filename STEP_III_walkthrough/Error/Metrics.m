actual = readmatrix("Actual.csv");

expected = readmatrix("Expected.csv");

diff = expected-actual;
binEdges = linspace(-9.5,9.5,40);

f = figure('Position',[10 10 600 300]);
hold on;
grid;
handle = histogram(diff,binEdges, "Normalization","probability");
xlabel("$$\textnormal{Error}\ [\cdot]$$", Interpreter="latex");
ylabel("$$\textnormal{Probability}\ [\cdot]$$", Interpreter="latex");
ylim([0 1]);
xlim([-2 2]);
hold off; 

exportgraphics(f,"histogram.pdf","Resolution",150);

%histogram(diff,5, "Normalization","probability")