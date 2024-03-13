from uflp import UFLP
from typing import List, Tuple
""" 
    Binome 1 : Nom Prenom (Matricule)
    Binome 2 : Nom Prenom (Matricule)
    Description succinte de l'implementation :
    ...
"""

def solve(problem: UFLP) -> Tuple[List[int], List[int]]:
    """
    Votre implementation, doit resoudre le probleme via recherche locale.

    Args:
        problem (UFLP): L'instance du probleme à résoudre

    Returns:
        Tuple[List[int], List[int]]: 
        La premiere valeur est une liste représentant les stations principales ouvertes au format [0, 1, 0] qui indique que seule la station 1 est ouverte
        La seconde valeur est une liste représentant les associations des stations satellites au format [1 , 4] qui indique que la premiere station est associée à la station pricipale d'indice 1 et la deuxieme à celle d'indice 4
    """
    # run the solver brute force
    n = problem.n_satellite_station
    k = problem.n_main_station
    main_station_opened = [0] * k
    association_index = [0] * n
    main_station_opened[0] = 1
    actual_cost = problem.calculate_cost(main_station_opened, association_index)

    for i in range(k):
        main_station_opened[i] = 1
        old_cost = actual_cost
        for j in range(n):
            old_index = association_index[j]
            association_index[j] = i
            new_cost = problem.calculate_cost(main_station_opened, association_index)
            if new_cost < actual_cost:
                actual_cost = new_cost
            else:
                association_index[j] = old_index
        if actual_cost == old_cost:
            main_station_opened[i] = 0
    return main_station_opened, association_index

if __name__ == "__main__":
    solve(UFLP("instance_A_4_6"))