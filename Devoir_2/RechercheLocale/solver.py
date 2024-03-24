from uflp import UFLP
from typing import List, Tuple
import random

"""
    Binome 1 : Renel Lherisson (2089776)
    Binome 2 : Daniel Giao (2120073)

    Description succinte de l'implementation :
    La fonction solve utilise une recherche locale avec redémarrages.
    Elle génère une solution initiale aléatoire
    et améliore ces solutions en inversant l'état des stations principales (ouvertes/fermées)
    et en recalculant les associations des stations satellites pour minimiser le coût.
    Le redémarrage permet de sortir des minimas locaux.
    Ainsi, la meilleure solution trouvée est retournée apres les redémarrages.
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

    # Nombre de redémarrage fixé à 10 au cas où on tombe dans un minimum local
    restart = 10
    # Boucle de redémarrage
    for _ in range(restart):
        # Génération d'une solution initiale aléatoire avec au moins une station principale ouverte
        main_station_opened = [random.randint(0, 1) for _ in range(k)]
        # Tester qu'au moins une station principale doit être ouverte, sinon la solution a calculer devient infaisable
        while sum(main_station_opened) == 0:
            main_station_opened = [random.randint(0, 1) for _ in range(k)]
        open_stations = [i for i, station in enumerate(main_station_opened) if station == 1]
        association_index = [random.choice(open_stations) for _ in range(n)]

        # Cout de la solution initiale générée
        actual_cost = problem.calculate_cost(main_station_opened, association_index)

        # Boucle d'amélioration de la solution initiale générée
        cost_improved = True
        # Tant que le coût de la solution est amélioré, on continue d'améliorer la solution
        while cost_improved:
            cost_improved = False
            # Amélioration de la solution
            main_station_opened, association_index, actual_cost, cost_improved = improve_solution(problem, main_station_opened, association_index, actual_cost, k, n)

        # Mise à jour de la meilleure solution trouvée après chaque redémarrage
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
    # Iterate over the main stations and update the associations
    for i in range(k):
        # Create a neighbor solution by changing the state with current main station opened
        main_station_opened_neighbor = main_station_opened[:]
        main_station_opened_neighbor[i] = 1 - main_station_opened_neighbor[i]
        # If at least one main station is opened, we update the associations
        if sum(main_station_opened_neighbor) > 0:
            # Find the association index for each satellite station based on the opened main stations
            association_index_neighbor = find_association_index(problem, main_station_opened_neighbor, n, k)
            # Calculate the cost of the new solution
            new_cost = problem.calculate_cost(main_station_opened_neighbor, association_index_neighbor)
            # If the new cost is less than the actual cost, we update the solution
            if new_cost < actual_cost:
                # Update the solution
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
    # Find the association index for each satellite station based on the opened main stations
    for satellite_station_index in range(n):
        min_cost, min_index = float('inf'), -1
        # Find the main station with the minimum association cost for the satellite station
        for main_station_index in range(k):
            # If the main station is opened, we calculate the association cost
            if main_station_opened[main_station_index] == 1:
                # Calculate the association cost
                cost = problem.get_association_cost(main_station_index, satellite_station_index)
                # Update the minimum cost and index
                if cost < min_cost:
                    min_cost, min_index = cost, main_station_index
        # Append the index of the main station with the minimum association cost
        association_index.append(min_index)

    return association_index
