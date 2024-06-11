def try_load(pomdp_factory, id):
    resources = "resources/cache/pomdps"
    try:
        from pomdp import PartiallyObservableMarkovDecisionProcess
        pomdp = PartiallyObservableMarkovDecisionProcess.load(f"{resources}/{id}")
    except FileNotFoundError:
        pomdp = pomdp_factory()
        pomdp.save(resources)
    return pomdp
