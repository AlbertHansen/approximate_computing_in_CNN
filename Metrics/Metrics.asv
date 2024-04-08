actual = readmatrix("expectedFile.csv");
actual = actual(1:end-1);
expected = readmatrix("actualFile.csv");

diff = expected-actual;

WCE = max(diff)


histogram(diff,50);