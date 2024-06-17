function isSubmatrix = checkSubmatrix(A, B)
    % Get dimensions of matrices A and B
    [rowsA, colsA] = size(A);
    [rowsB, colsB] = size(B);

    % Initialize flag
    isSubmatrix = false;

    % Check if dimensions of A are less than or equal to those of B
    if rowsA > rowsB || colsA > colsB
        return;
    end

    % Slide A over B and check for matches
    for i = 1:(rowsB - rowsA + 1)
        for j = 1:(colsB - colsA + 1)
            % Extract submatrix of B of the same size as A
            subB = B(i:i+rowsA-1, j:j+colsA-1);
            
            % Check if the submatrix matches A
            if isequal(subB, A)
                isSubmatrix = true;
                return;
            end
        end
    end
end
