% RENEL LHERISSON (2089776)
% DANIEL GIAO (2120073)

include "globals.mzn";

% Nombre de fréquences à placer
int: m;
% Placements possibles selon le nombre de fréquences
int: n = m*m;

% Définition de la plage de fréquences 
array[1..m] of var 0..n: frequences;

% Calcul des distances entre les différentes fréquences
array[1..(m*(m-1)) div 2] of var 0..n: distances =
    [ frequences[j] - frequences[i] | i in 1..m, j in i+1..m];

% Variable objectif à minimiser
var int: longueur;

%-----------------------------------------------------------------------------%
% Partie étudiant
%-----------------------------------------------------------------------------%

% Contrainte pour garantir que la premiere frequence est 0
constraint frequences[1] = 0;

% Contrainte pour garantir que les fréquences sont placées dans l'ordre croissant
constraint increasing(frequences);

% Contrainte pour garantir que toutes les fréquences sont réparties à des distances
% différentes les unes des autres
constraint alldifferent(distances);

% Contrainte pour définir la longueur totale de la plage
constraint longueur = max(frequences);

% Contrainte pour limiter la longueur de la plage
constraint longueur <= sum([abs(distances[i]) | i in 1..(m*(m-1)) div 2]);

%-----------------------------------------------------------------------------%

output 
  [show(frequences)]
  ++
  ["\nLongueur de la plage = ", show(longueur)];
