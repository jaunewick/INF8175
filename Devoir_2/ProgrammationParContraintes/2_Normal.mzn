% RENEL LHERISSON (2089776)
% DANIEL GIAO (2120073)

include "globals.mzn";

% États possibles pour une case du terrain
% V = Vide
% B = Tour bleue
% R = Tour rouge
enum etat = {V, B, R};

% Dimension du terrain (carré)
int: n;
set of int: dim = 1..n;

% État du terrain
array[dim, dim] of var etat: terrain;

% Variable objectif à maximiser
var int: total_commun;

%-----------------------------------------------------------------------------%
% Partie étudiant
%-----------------------------------------------------------------------------%

% Contrainte pour garantir le même nombre de tours
constraint count(terrain, B) == count(terrain,R);

% Contrainte pour garantir qu'il n'y ait aucun croisement horizontal
constraint forall(i in dim) (
  not exists(j in dim) (
    terrain[i, j] == B /\ exists(k in dim) (
      terrain[i, k] == R
    )
  )
);

% Contrainte pour garantir qu'il n'y ait aucun croisement vertical
constraint forall(i in dim) (
  not exists(j in dim) (
    terrain[j, i] == B /\ exists(k in dim) (
      terrain[k, i] == R
    )
  )
);

% Contrainte pour garantir qu'il n'y ait aucun croisement diagonal
constraint forall(i in 1..n, j in 1..n where terrain[i, j] == B) (
  forall(k in 1..n, l in 1..n where terrain[k, l] == R) (
    abs(i - k) != abs(j - l)
  )
);

% Contrainte pour fixer une configuration initiale ne pouvant pas être symétrique, e.g tour R à la position (1,1)
constraint terrain[1,1] = R;

% Maximiser total_comun
total_commun = count(terrain, R);
solve maximize total_commun;
%-----------------------------------------------------------------------------%
output 
["Terrain :\n"]
++
[if j = 1 then "["  endif ++ if fix(terrain[i,j]) == V then "_" else show(terrain[i,j]) endif ++ if j = n then "]\n" else "," endif | i in 1..n, j in 1..n]
++
["\n"]
++
["Total de tours commun: " ++ show(total_commun)]
