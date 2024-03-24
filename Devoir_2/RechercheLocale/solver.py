from uflp import UFLP
from typing import List, Tuple
import random

"""
    Binome 1 : Renel Lherisson (2089776)
    Binome 2 : Daniel Giao (2120073)

    Description succinte de l'implementation :
    La fonction solve utilise une recherche locale avec redémarrages.
    Elle génère des solutions initiales aléatoires
    et améliore ces solutions en inversant l'état des stations principales (ouvertes/fermées)
    et en recalculant les associations des stations satellites pour minimiser le coût.
    Le redémarrage permet de s'en sortir des minimas locaux.
    Ainsi, la meilleure solution trouvée est retournée au cours de tous les redémarrages.
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
        La seconde valeur est une liste représentant les associations des stations js au format [1 , 4] qui indique que la premiere station est associée à la station pricipale d'indice 1 et la deuxieme à celle d'indice 4
    """

    # n : Nombre de stations satellites
    n = problem.n_satellite_station
    # k : Nombre de stations principales
    k = problem.n_main_station

    # Initialisation
    best_solution = ()
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

        # Boucle d'amélioration de la solution initiale
        cost_improved = True
        while cost_improved:
            cost_improved = False
            main_station_opened, association_index, actual_cost, cost_improved = improve_solution(problem, main_station_opened, association_index, actual_cost, k, n)

        # Mise à jour de la meilleure solution trouvée
        if actual_cost < best_cost:
            best_cost = actual_cost
            best_solution = (main_station_opened, association_index)

    return best_solution


def improve_solution(problem: UFLP, main_station_opened: List[int], association_index: List[int], actual_cost: float, k: int, n: int) -> Tuple[List[int], List[int], float, bool]:
    """
    Improve the solution by iterating over the main stations and updating the associations.

    Args:
        problem (UFLP): The problem instance.
        main_station_opened (List[int]): List representing the opened main stations.
        association_index (List[int]): List representing the associations of satellite stations.
        actual_cost (float): The current cost of the solution.
        k (int): Number of main stations.
        n (int): Number of satellite stations.

    Returns:
        Tuple[List[int], List[int], float, bool]: The updated main_station_opened, association_index, actual_cost, and cost_improved.
    """
    cost_improved = False
    for i in range(k):
        main_station_opened_neighbor = main_station_opened[:]
        main_station_opened_neighbor[i] = 1 - main_station_opened_neighbor[i]

        if sum(main_station_opened_neighbor) > 0:
            association_index_neighbor = find_association_index(problem, main_station_opened_neighbor, n, k)
            new_cost = problem.calculate_cost(main_station_opened_neighbor, association_index_neighbor)
            if new_cost < actual_cost:
                main_station_opened, association_index = main_station_opened_neighbor, association_index_neighbor
                actual_cost = new_cost
                cost_improved = True
                break

    return main_station_opened, association_index, actual_cost, cost_improved

def find_association_index(problem: UFLP, main_station_opened: List[int], n: int, k: int) -> List[int]:
    """
    Find the association index for each satellite station based on the opened main stations.

    Args:
        problem (UFLP): The problem instance.
        main_station_opened (List[int]): List representing the opened main stations.
        n (int): Number of satellite stations.
        k (int): Number of main stations.

    Returns:
        List[int]: The association index for each satellite station.
    """
    association_index = []
    for satellite_station_index in range(n):
        min_cost, min_index = float('inf'), -1
        for main_station_index in range(k):
            if main_station_opened[main_station_index] == 1:
                cost = problem.get_association_cost(main_station_index, satellite_station_index)
                if cost < min_cost:
                    min_cost, min_index = cost, main_station_index
        association_index.append(min_index)

    return association_index
