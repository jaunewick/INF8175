from uflp import UFLP
from typing import List, Tuple
import random

""" 
    Binome 1 : Renel Lherisson (2089776)
    Binome 2 : Daniel Giao (2120073)
    Description succinte de l'implementation :
    ...
"""

def solve_brute_force(problem: UFLP) -> Tuple[List[int], List[int]]:
    """
    Votre implementation, doit resoudre le probleme via recherche locale.

    Args:
        problem (UFLP): L'instance du probleme à résoudre

    Returns:
        Tuple[List[int], List[int]]: 
        La premiere valeur est une liste représentant les stations principales ouvertes au format [0, 1, 0] qui indique que seule la station 1 est ouverte
        La seconde valeur est une liste représentant les associations des stations js au format [1 , 4] qui indique que la premiere station est associée à la station pricipale d'indice 1 et la deuxieme à celle d'indice 4
    """
    # run the solver brute force
    n = problem.n_j_station
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


def solve(problem: UFLP) -> Tuple[List[int], List[int]]:
    """
    Votre implementation, doit resoudre le probleme via recherche locale.

    Args:
        problem (UFLP): L'instance du probleme à résoudre

    Returns:
        Tuple[List[int], List[int]]: 
        La premiere valeur est une liste représentant les stations principales ouvertes au format [0, 1, 0] qui indique que seule la station 1 est ouverte
        La seconde valeur est une liste représentant les associations des stations js au format [1 , 4] qui indique que la premiere station est associée à la station pricipale d'indice 1 et la deuxieme à celle d'indice 4
    """

    # n : Nombre de stations satellites
    n = problem.n_satellite_station
    # k : Nombre de stations principales
    k = problem.n_main_station

    # Initialisation
    best_solution = (main_station_opened := [], association_index := [])
    best_cost = float('inf')

    # Nombre de redémarrage
    restart = 10
    for _ in range(restart):
        # Génération d'une solution initiale aléatoire
        main_station_opened = [random.randint(0, 1) for _ in range(k)]
        # Au moins une station principale doit être ouverte, sinon infaisable
        while sum(main_station_opened) == 0:
            main_station_opened = [random.randint(0, 1) for _ in range(k)]
        open_stations = [i for i, station in enumerate(main_station_opened) if station == 1]
        association_index = [random.choice(open_stations) for _ in range(n)]

        # Cout de la solution initiale
        actual_cost = problem.calculate_cost(main_station_opened, association_index)

        # Boucle d'amélioration
        cost_improved = True
        while cost_improved:
            cost_improved = False
            for i in range(k):
                # Définition du voisinage :
                # Idée : Générer un ensemble de solutions voisines égal au nombre de stations principales
                main_station_opened_neighbor = main_station_opened[:]
                # Inversion de l'état de la station principale
                main_station_opened_neighbor[i] = 1 - main_station_opened_neighbor[i]

                if sum(main_station_opened_neighbor) > 0:
                    association_index_neighbor = []
                    for satellite_station_index in range(n):
                        min_cost, min_index = float('inf'), -1
                        for main_station_index in range(k):
                            # Définition des voisins valides
                            if main_station_opened_neighbor[main_station_index] == 1:
                                cost = problem.get_association_cost(main_station_index, satellite_station_index)
                                if cost < min_cost:
                                    min_cost, min_index = cost, main_station_index
                        # Sélection d'un voisin
                        association_index_neighbor.append(min_index)

                    # Mise à jour de la meilleure solution trouvée
                    new_cost = problem.calculate_cost(main_station_opened_neighbor, association_index_neighbor)
                    if new_cost < actual_cost:
                        main_station_opened, association_index = main_station_opened_neighbor, association_index_neighbor
                        actual_cost = new_cost
                        cost_improved = True
                        break
        
        # Mise à jour de la meilleure solution trouvée
        if actual_cost < best_cost:
            best_cost = actual_cost
            best_solution = (main_station_opened, association_index)

    return best_solution

if __name__ == "__main__":
    solve(UFLP("instance_A_4_6"))