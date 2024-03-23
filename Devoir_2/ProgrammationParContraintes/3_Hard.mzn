% RENEL LHERISSON (2089776)
% DANIEL GIAO (2120073)

include "globals.mzn";

% Énumérations des équipes participant au tournoi
enum equipes = {CO, TA, HO, DA, NA, MU, AL, UT};

% Lieux de rencontres des equipes (0 = pas de rencontre).
% Tout lieu i, tel que 0 < i <= card(equipes), 
% est designé comme le domicile de l'équipe associée par la relation to_enum(equipe, i)
set of int: terrains = 0..card(equipes);

% Étendue du tournoi (en jours)
int: d = 16;
set of int: jours = 1..d;

% Calendrier de placement des matchs
array[equipes, jours] of var terrains: calendrier;

% Nombres de rencontres entre deux equipes
array[equipes, equipes] of var 0..2: rencontres;

% Tuples des dates de match entre deux équipes
array[equipes, equipes] of var tuple (0..d, 0..d): dates;

% Variable objectif à maximiser
var int: objectif_match_espaces;

%-----------------------------------------------------------------------------%
% Partie étudiant
%-----------------------------------------------------------------------------%

% Contrainte 1: Deux équipe doivent être présente sur le même terrain.
constraint forall(e1, e2 in equipes where e1 < e2) (
    (calendrier[e1, dates[e1, e2].1] = calendrier[e2, dates[e1, e2].1]) /\
    (calendrier[e1, dates[e1, e2].2] = calendrier[e2, dates[e1, e2].2])
);

% Contrainte 2: Un match doit avoir lieu sur le terrain désigné comme le domicile de l’une des deux équipes.
constraint forall(e1, e2 in equipes where e1 < e2) (
    (calendrier[e1, dates[e1, e2].1] = e1 \/ calendrier[e1, dates[e1, e2].1] = e2) /\
    (calendrier[e1, dates[e1, e2].2] = e1 \/ calendrier[e1, dates[e1, e2].2] = e2)
);

% Contrainte 3: Un terrain ne peut accueillir plus de deux équipes.
constraint forall(j in jours, t in terrains where t > 0) (
    sum(e in equipes) (calendrier[e, j] = t) <= 2
);

% Contrainte 4: Si une équipe ne joue pas lors d’une journée, elle est placée sur le terrain n°0.
constraint forall(e in equipes, j in jours) (
    calendrier[e, j] >= 0
);

% Contrainte 5: Une équipe ne doit pas jouer trop souvent à domicile, elle n’est donc pas autorisée à jouer plus de 7 jours sur son propre terrain.
constraint forall(e in equipes) (
    count([calendrier[e, j] | j in jours], e) <= 7
);

% Contrainte 6: Les équipes doivent jouer deux fois l’une contre l’autre.
constraint forall(e1, e2 in equipes where e1 < e2) (
    rencontres[e1, e2] = 2
);

% Contrainte 7: Les deux rencontres de deux équipes doivent être espacées de 3 jours au minimum. 
constraint forall(e1, e2 in equipes where e1 < e2) (
    abs(dates[e1, e2].2 - dates[e1, e2].1) >= 3
);

% Contrainte 8: Les deux rencontres de deux équipes doivent avoir lieu sur des terrains différents.
constraint forall(e1, e2 in equipes where e1 < e2) (
    calendrier[e1, dates[e1, e2].1] != calendrier[e1, dates[e1, e2].2]
);

% Contrainte symétrique pour que le premier match entre chaque paire d'équipes a lieu sur le terrain de e1
constraint forall(e1, e2 in equipes where e1 < e2)(
    calendrier[e1, dates[e1, e2].1] = e1
);

% Maximiser objectif_match_espaces
% i.e. le nombre de fois où deux matchs entre les mêmes équipes sont espacés par aux moins 6 jours
objectif_match_espaces = sum(e1, e2 in equipes where e1 < e2) (
    abs(dates[e1, e2].2 - dates[e1, e2].1)  >=  6
);
solve maximize objectif_match_espaces;

%-----------------------------------------------------------------------------%

% Affichage du calendrier par equipes et par jours puis du résultat de la fonction objectif.
% Lorsqu'une équipe ne joue pas, l'emplacement de match reste vide sinon elle contient le nom de l'équipie adverse.
% Si une équipe ne joue pas à domicile, un "@" précède le nom de l'équipe adverse.
output 
["Calendrier:\n"]
++
[if j = 1 then "   " else "" endif ++ if j < 10 then "  j" ++ show(j) else " j" ++ show(j) endif ++ if j = d then "\n" else "" endif | j in jours ]
++
[if j = 1 then show(e1) ++ ":" else "" endif ++ if fix(calendrier[e1,j]) = 0 then "    " else if fix(calendrier[e1,j]) = e1 then "  " ++ concat([if e1 != e2 /\ fix(calendrier[e2,j]) = e1 then show(e2) else "" endif | e2 in equipes])  else " @" ++ show(to_enum(equipes, calendrier[e1,j])) endif endif ++ if j = d then "\n" else "" endif | e1 in equipes, j in jours ]
++
["\nMatchs retour espacés de 6 jours ou plus = " ++ show(objectif_match_espaces) ++ "\n"]
;
