from backend.environments import pursuit, overcooked, gridworld, cmu_gridworlds, predator_prey
from pomdp import PartiallyObservableMarkovDecisionProcess

factory = {

    #"gridworld3x3": lambda model_id, noise: gridworld.create(model_id, noise, (3, 3), name="gridworld3x3"),
    #"gridworld4x4": lambda model_id, noise: gridworld.create(model_id, noise, (4, 4), name="gridworld4x4"),

    #"gridworld5x5": lambda model_id, noise: gridworld.create(model_id, noise, (5, 5), name="gridworld5x5"),

    #"pursuit3x3": lambda model_id, noise: pursuit.create(model_id, noise, (3, 3), name="pursuit3x3"),
    #"pursuit4x4": lambda model_id, noise: pursuit.create(model_id, noise, (4, 4), name="pursuit4x4"),
    #"pursuit5x5": lambda model_id, noise: pursuit.create(model_id, noise, (5, 5), name="pursuit5x5"),
    #"pursuit6x6": lambda model_id, noise: pursuit.create(model_id, noise, (6, 6), name="pursuit6x6"),
    #"nju": lambda model_id, noise: cmu_gridworlds.create(model_id, noise, "nju"),

    "predator-prey": lambda model_id, comms: predator_prey.create(model_id, comms, (15, 15)),

    "debugworld": lambda model_id, noise: gridworld.create(model_id, noise, (3, 3), name="debugworld"),

    "gridworld": lambda model_id, noise: gridworld.create(model_id, noise, (5, 5), name="gridworld"),
    "pursuit-task": lambda model_id, comms: pursuit.create(model_id, comms, (5, 5), name="pursuit-task"),
    "pursuit-teammate": lambda model_id, noise: pursuit.create(model_id, noise, (5, 5), name="pursuit-teammate"),
    "pursuit-both": lambda model_id, noise: pursuit.create(model_id, noise, (5, 5), name="pursuit-both"),

    "overcooked": lambda model_id, noise: overcooked.create(model_id, noise, name="overcooked"),

    "ntu": lambda model_id, noise: cmu_gridworlds.create(model_id, noise, "ntu"),
    "isr": lambda model_id, noise: cmu_gridworlds.create(model_id, noise, "isr"),
    "mit": lambda model_id, noise: cmu_gridworlds.create(model_id, noise, "mit"),
    "pentagon": lambda model_id, noise: cmu_gridworlds.create(model_id, noise, "pentagon"),
    "cit": lambda model_id, noise: cmu_gridworlds.create(model_id, noise, "cit"),
}
