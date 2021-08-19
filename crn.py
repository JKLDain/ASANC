import numpy as np

from crn import Species, Simulation, utils
from numpy import log
from numpy.random import choice
from random import random, uniform
from scipy.integrate import odeint

with utils.no_output():
    import stochpy

class CRN:
    """ A Chemical Reaction Network (CRN)

    args(인자):
        *system: List[Reaction]
            The chemical reactions that define the system.
            시스템을 정의하는 chemical reaction
    attributes:
        system: List[Reaction]
            The chemical reactions that define the system.
        species: Set[Species]
            The set of species string names that are present in the CRN.
            CRN에 존재하는 species 문자열 이름 set
        kwargs: Dict[]
            name: str
                defaults to `id(self)`. Used for naming any files produced
                by the CRN during simulation.
                기본값: 'id(self)'
                시뮬레이션동안 CRN에서 생성된 파일의 이름 지정에 사용
    """

    def __init__(self, *system, **kwargs):
        self.system = system
        self.species = self.get_species()
        self.species_index = self.get_species_index()
        self.reactions_index = self.get_reactions_index()
        self.diffeq_system_func = self.rate_laws()
        self.name = kwargs.get("name", id(self))

    def get_species(self):
        """
        Returns the set of species present in the CRN.
        CRN에 있는 species 세트 리턴
        """
        species = set()
        for s in self.system:
            species |= s.get_species()
        return species

    def get_species_index(self):
        """
        Get a map of int to species, which is unchanged for the lifetime
        of the CRN.
        CRN의 수명 동안 변경되지 않은 int에서 species까지의 map을 가져옴
        This is meant for internal use, and won't be very useful for anyone
        using the CRN.
        내부적인 사용일뿐 CRN을 사용하는 사람에게 유용한 건 아님
        """
        return dict(enumerate(sorted(self.species)))

    def get_reactions_index(self):
        """
        Get a map of int to reactions, which is unchanged for the lifetime
        of the CRN.
        CRN 동안 변경되지 않은 int에서 reaction까지의 map을 가져옴

        This is meant for internal use, and won't be very useful for anyone
        using the CRN.
        """
        return dict(enumerate(self.system))

    def rate_law_for_species(self, s):
        """
        Returns the symbolic representation for the rate law of species `s`.
        species 's'의 비율 법칙을 위한 기호 표현 리턴

        args:
            s: Union[Species, str]
                the species whose rate law in the CRN `self` is desired.
                `s` can either be an instance of the Species object or
                the string name of the Species.
                CRN의 rate law를 원하는 species.
                's'는 species 개체의 instance 이거나 Species의 문자열 이름일 수 있음
        """
        if type(s) not in (str, Species):
            raise ValueError("rate_law_for_species called on non-species. "
                             "parameter must be a species name (str) or a "
                             "Species instance.")

        if type(s) is Species:
            s = s.name

        return sum(rxn.net_production(s) * rxn.flux() for rxn in self.system)

    def rate_laws(self):
        """
        Returns a function that takes a list of species concentrations,
        in the same order as specified in `self.species_index`, and returns
        the a vector of the current rate of change of each species, again in
        the same order as specified in `self.species_index`.
        'self.species_index'에 지정된 순서와 동일한 'species concentrations' 리스트를 가져오고
        현재 각 species의 변화율에 대한 벡터를 반환한다.
        """
        laws = []

        # Has to be iterated this way to laws is in order
        # 이런식으로 반복해야 law가 성립됨
        for i in range(len(self.species_index)):
            sp = self.species_index[i]
            law = self.rate_law_for_species(sp)
            for i, s in self.species_index.items():
                sub = 1 if s == "nothing" else f"v[{i}]"
                law = law.subs(s, sub)
            laws.append(str(law))

        func = f"lambda v, t: [{', '.join(laws)}]"
        return eval(func)

    def stoch_simulate(self, amounts, t=20):
        """
        Stochastic discrete simulation of the CRN until time `t` with initial
        molecule count `amounts`. The species that are omitted from the
        dictionary of initial concentrations are assumed to have an initial
        count of 0. The simulation runs until time `t` unless the
        propensities all reach zero. If that's the case, the system has
        reached a steady state, and the simulation stops.

        초기 molecule count 'amount'로 시간 't'까지의 CRN의
        Stochastic discrete simulation(확률적 이산 시뮬레이션)
        초기 concentrations dictionary에서 생략된 species는 초기 count가 0인 것으로 가정.
        모든 propensity가 0에 도달하지 않는 한 시뮬레이션은 시간 't'까지 실행됨.

        args:
            amount: Dict[Species, int]
                A map describing each species' initial count.
                각 species의 초기 count를 나타내는 map
            t: Union[float, int]
                The upper bound of the time to run the simulation to.
                시뮬레이션 실행 시간의 상한
        """

        # Write psc file
        pscfile = utils.datadir(f"{self.name}.psc")
        self.write_pscfile(pscfile, amounts)

        # invoke stochpy
        smod = stochpy.SSA()
        smod.Model(pscfile)
        smod.DoStochSim(mode="time", end=t)

        # return data
        data = {}
        for sp in self.species:
            if sp.name != "nothing":
                data[sp] = smod.data_stochsim.getSimData(sp.name)[:, 1]
                if "time" not in data:
                    data["time"] = smod.data_stochsim.getSimData(sp.name)[:, 0]

        return Simulation(data, stochastic=True)

    def write_pscfile(self, filename, amounts):
        """
        Write the CRN in PySCeS Model Description Language for stochastic
        discrete simulation to `filename`.
        stochastic discrete simulation을 위해 CRN을 'PySCeS' 모델 설명 언어로 '파일 이름'에 작성

        args:
            filename: str
                name of the .psc file to be written
                작성할 .psc 파일의 이름
            amounts: Dict[Species, int]
                A map describing each species' initial count.
                각 species의 초기 count에 대한 map
        """

        def fmt_expression(expr):
            formatted_expr = []
            for species, c in expr.species.items():
                # special PySCeS Model Description Language Syntax
                # 특수 PySCS 모델 설명 언어 구문
                species = species.name
                if species == "nothing":
                    species = "$pool"

                if c != 1:
                    formatted_expr.append(f"{{{c}}}{species}")
                else:
                    formatted_expr.append(species)

            return " + ".join(formatted_expr)

        with open(filename, "w") as pscfile:
            pscfile.write("# Reactions\n\n")
            for i, rxn in self.reactions_index.items():
                pscfile.write(f"R{i}:\n")
                reac = fmt_expression(rxn.reactants)
                prod = fmt_expression(rxn.products)
                pscfile.write(f"{reac} > {prod}\n")
                pscfile.write(f"k{i}*{rxn.discrete_flux()}\n\n")
            pscfile.write("\n")

            pscfile.write("# Rate constants\n")
            for i, rxn in self.reactions_index.items():
                pscfile.write(f"k{i} = {rxn.coeff}\n")

            pscfile.write("\n# Initial Species Counts\n")

            # The `amounts` dictionary can have keys of type Species, so
            # we convert everything to the string names of the species.
            # 'amounts' dict에 Species 타입의 키가 있을 수 있으므로
            # 모든 species의 문자열 이름으로 변환
            amounts = amounts.copy()

            for sp in self.species:
                if sp not in amounts:
                    amounts[sp] = 0

            if Species("nothing") in amounts:
                del amounts[Species("nothing")]

            for sp in self.species:
                if sp.name != "nothing":
                    pscfile.write(f"{sp} = {amounts.get(sp, 0)}\n")


    def simulate(self, conc, t=20, resolution=100):
        """
        Deterministic concentration-continuous simulation of the CRN until
        time t with initial concentrations `conc`.
        The species that are omitted from the dictionary of initial
        concentrations are assumed to have an initial concentration of 0.0.
        초기 concentrations 'conc'로 시간 t까지의 결정론적 CRN의 연속 시뮬레이션.
        초기 concentration의 dict에서 생략된 species는 초기 concentration이 0.0인 것으로 가정

        args:
            conc: Dict[Species, float]
                A map describing each species' initial concentration.
            t: Union[float, int]
                The upper bound of the time to run the simulation to.
            resolution: int
                How many time steps to simulate between times [0, t).
                [0, t)의 시간동안 시뮬레이션할 시간 단계 수
        """

        t = np.linspace(0, t, resolution)

        conc_temp = {}

        for s, c in conc.items():
            if type(s) not in (str, Species):
                raise ValueError("CRN.simulate: initial concentrations "
                                 f"dictionary got a key with type {type(s)}. "
                                 "Key type should be Species or str.")

            if type(s) is Species:
                s = s.name

            conc_temp[s] = c

        conc = conc_temp

        v0 = [0] * len(self.species)

        for i, s in self.species_index.items():
            v0[i] = conc.get(s, 0)

        sol = odeint(self.diffeq_system_func, v0, t)

        sol_dict = {'time': t}

        for i, s in self.species_index.items():
            sol_dict[s] = sol[:, i]

        return Simulation(sol_dict)

    def schema_simulate(self, initial_counts, time=None, steps=None):
        """
        Stochastic simulator for reaction schema.
        reaction shcema를 위한 확률적 시뮬레이터
        """
        def possible_reactions(state):
            rxns = []
            for rxn in self.system:
                if rxn.is_schema:
                    for matching_rxn in rxn.possible_reactions(state):
                        rxns.append(matching_rxn)
                rxns.append(rxn)
            return rxns

        for sp in initial_counts:
            if sp.has_groups():
                raise ValueError(
                        "Schema with unmatched groups passed in to "
                        "'initial_counts'. Did you "
                        "forget to nullify a Schema's named groups? If "
                        "`schema` is a variable containing a species "
                        "returned from schemas(...), make sure to do "
                        "`{schema(): count}` and not `{schema: count}` when "
                        "passing as the initial counts of species to "
                        "'schema_simulate'.")


        if time is None and steps is None:
            time = float("inf")
            steps = 1000
        elif time is None:
            time = float("inf")
        elif steps is None:
            steps = float("inf")
        else:
            raise ValueError(
                    "CRN.schema_simulate: both 'time' and 'steps' were "
                    "passed. Pass in one or the other")


        curr_time = curr_steps = 0
        state = {sp: count for sp, count in initial_counts.items() if count}

        sim = {}
        history = {}
        sim["time"] = [0]
        sim["reactions"] = []
        for sp, count in state.items():
            sim[sp] = [count]

        # TODO (enricozb): add species history


        while True:
            if curr_time >= time or curr_steps >= steps:
                break

            # Grab reactions and propensities
            # reaction과 propensity(경향)를 파악
            rxns = possible_reactions(state)
            props = [rxn.propensity(state) for rxn in rxns]

            # Pick reaction to occur
            # 발생할 reaction을 선택함
            p_tot = sum(props)

            if p_tot == 0:
                print("simulation ended before reaching 'time' or 'steps'")
                break

            rxn = choice(rxns, p=[p / p_tot for p in props])
            sim["reactions"].append(rxn)

            # Pick dt and register chosen reaction effects
            # dt를 선택하고 선택한 reaction effects를 등록
            for r, c in rxn.reactants.species.items():
                state[r] -= c
                sim[r].append(state[r])
                if state[r] == 0:
                    del state[r]

            for p, c in rxn.products.species.items():
                if p not in state:
                    state[p] = 0
                    sim[p] = [0] * (curr_steps + 1)
                state[p] += c
                sim[p].append(state[p])

            dt = -log(uniform(0, 1)) / p_tot
            curr_time += dt
            curr_steps += 1

        return Simulation(sim)


    def validate(self, func, *, input_species, output_species, N=100,
            eps=1e-2, t=500):
        """
        Determine if the CRN `self` actually describes the computation in
        function `func`. This is a probabalistic verification: it runs
        several simulations at several different initial concentrations,
        then computes the desired function output and checks if the CRNs
        simulation is within `eps` of `func`s output.
        CRN 'self'가 실제로 함수 'func'에서 계산을 설명하는지 확인함.
        이것은 확률론적 검증임. 여러 초기 concentration에서 여러 시뮬레이션을 실행한 다음
        원하는 함수 출력을 계산하고 CRNs 시뮬레이션이 'func' 출력의 'eps' 이내에 있는지 확인.

        args:
            func: Callable[Dict[Expression, float], float]
                This function takes in the initial concentrations of the
                input species to the CRN and outputs a number computed from
                these initial concentrations.
                이 함수는 입력 species의 초기 concentration을 CRN으로 가져와서
                이러한 초기 concentration에서 계산한 숫자를 출력함.
        """
        for i in range(N):
            species = {sp : random() * 10 for sp in input_species}
            theoretical = func(species)
            sim = self.simulate(species, t=t)
            simulated = sim[output_species][-1]

            if abs(theoretical - simulated) > eps:
                return {"success": False, "sim": sim, **species,
                        "theoretical": theoretical, "simulated": simulated}

        return {"success": True}

