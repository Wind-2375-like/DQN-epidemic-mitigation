""" Handles the epidemic modeling and the parsing of a yaml configuration file.

    This class doesn't "know" that it's an RL environment, this part is handled  by the wrapper class Env or DistributedEnv which handles the OpenAI gym part of the task.
"""

import yaml
import networkx as nx
import numpy as np
import random as rd
from typing import Tuple, Dict, List, Any
from dataclasses import dataclass

@dataclass
class Parameters():
    """Contains simulation parameters for either a city or the country on a given day
    """
    day: int
    suceptible: int
    exposed: int
    infected: int
    recovered: int
    dead: int
    initial_population: int

@dataclass
class Observables():
    """Observation class, contains all observable variables from either the city or the country
    """
    infected: List[int]
    dead: List[int]

@dataclass
class Observation():
    """Observation class, contains all fields returned when observing the environment
    """
    pop: Dict[str,float] # Contains the original population of each city
    city: Dict[str, Observables]
    total: Observables


class ModelDynamics():
    
    parameters = [  'infected',
                    'dead',
                    'suceptible',
                    'exposed',
                    'recovered',
                    'initial_population']

    def __init__(self, source_file:str):
        """Initializes the ModelDynamics class, creates a graph and sets epidemic dynamics  parameters from a source yaml file.

        Args:
            source_file (str): path to yaml initialization file
        """
        # loading the parameters from the yaml file
        doc = open(source_file, 'r')
        _params = yaml.safe_load(doc)
        try:
            # simulation parameters
            self.alpha = _params['alpha']
            self.var_alpha = _params['var_alpha']
            self.beta = _params['beta']
            self.var_beta = _params['var_beta']
            self.eta = _params['eta']
            self.var_eta = _params['var_eta']
            self.gamma = _params['gamma']
            self.var_gamma = _params['var_gamma']
            self.zeta = _params['zeta']
            self.var_zeta = _params['var_zeta']
            self.tau_0 = _params['tau_0']
            self.var_tau_0 = _params['var_tau_0']
            self.dt = _params['dt']

            # action parameters
            self.confinement_effectiveness = _params['confinement_effectiveness']
            self.isolation_effectiveness = _params['isolation_effectiveness']
            self.extra_hospital_effectiveness = _params['extra_hospital_effectiveness']
            self.vaccination_effectiveness = _params['vaccination_effectiveness']
            self.env_step_length = _params['env_step_lenght']
            self.srate = _params['srate']

            # cities and roads lists
            self.cities = list(_params['cities'].keys())
            self.n_cities = len(self.cities)
            if _params['roads'] is not None:
                self.roads = _params['roads']
            else:
                self.roads = []

            # generating a graph from the roads and cities
            self.map = nx.Graph()
            self.map.add_nodes_from(self.cities)
            self.map.add_edges_from(self.roads)

            self.pos_map = {}
            for c in self.cities:
                self.map.nodes[c]['pop'] = _params['cities'][c]['pop']
                self.pos_map[c] = [_params['cities'][c]
                                   ['lat'], _params['cities'][c]['lon']]

            self.NULL_ACTION = {'confinement': {c: False for c in self.cities},
                                'isolation': {c: False for c in self.cities},
                                'hospital': {c: False for c in self.cities},
                                'vaccinate': False,
                                }
            self.ACTION_CARDINALITY = len(self.NULL_ACTION)

        except:
            raise("Invalid YAML scenario file")

        self.total_pop = np.sum([self.map.nodes[n]['pop']
                                for n in self.map.nodes()])
        for e in self.roads:
            tau = 10*(self.map.nodes[e[0]]['pop'] *
                      self.map.nodes[e[1]]['pop'])/self.total_pop**2
            self.map.edges[e]['tau'] = tau

        self.reset()

    def reset(self):
        """Resets the dynamical system variables and control parameters
        """
        # initializing the variables
        nx.set_node_attributes(self.map, 1., "s")
        nx.set_node_attributes(self.map, 0., "e")
        nx.set_node_attributes(self.map, 0., "i")
        nx.set_node_attributes(self.map, 0., "r")
        nx.set_node_attributes(self.map, 0., "d")

        # initializing the control parameters
        self.c_confined = {c: 1 for c in self.cities}
        self.c_isolated = {c: 1 for c in self.cities}
        self.extra_hospital_beds = {c: 1 for c in self.cities}
        self.vaccinate = {c: 0 for c in self.cities}

    

    def draw_map(self,):
        """ Draws the map on which the epidemic is simulated (as a matplotlib plot).
        """
        nx.draw_networkx(self.map,
                with_labels=True,
                pos=self.pos_map,
                node_size=[self.map.nodes[n]['pop'] /
                           1000 for n in self.map.nodes()],
                width=[self.map.edges[e]['tau']*10 for e in self.map.edges()]
                )

    def epidemic_parameters(self, day=None)->Tuple[Dict[str,float],Dict[str,Dict[str,float]]]:
        """ Returns the state of the epidemic propagation.
        
        Args:u
            day (int): day at which the measurement is taken.

        Returns:
            Tuple[Dict[str,float],Dict[str,Dict[str,float]]]: a tuple containing
            
                in element 0: a dict containing the total suceptible, infected, recovered and dead population
                
                in element 1: a dict containing the suceptible, infected, recovered and dead population per city
        """
        cities = {}
        suceptible_total = 0
        exposed_total = 0
        infected_total = 0
        recovered_total = 0
        dead_total = 0
        total = 0

        for c in self.cities:
            suceptible = np.max(int(
                np.floor(self.map.nodes[c]['s'] * self.map.nodes[c]['pop'])),0)
            suceptible_total += suceptible
            exposed = np.max(int(
                np.floor(self.map.nodes[c]['e'] * self.map.nodes[c]['pop'])),0)
            exposed_total += exposed
            infected =np.max(int(
                np.floor(self.map.nodes[c]['i'] * self.map.nodes[c]['pop'])),0)
            infected_total += infected
            recovered = np.max(int(
                np.floor(self.map.nodes[c]['r'] * self.map.nodes[c]['pop'])),0)
            recovered_total += recovered
            dead = np.max(int(
                np.floor(self.map.nodes[c]['d'] * self.map.nodes[c]['pop'])))
            dead_total += dead
            total += self.map.nodes[c]['pop']

            cities[c] = Parameters(
                day=day,
                suceptible=suceptible,
                exposed=exposed,
                infected=infected,
                recovered=recovered,
                dead=dead,
                initial_population=self.map.nodes[c]['pop'],
            )

        total = Parameters(
            day=day,
            suceptible=suceptible_total,
            exposed=exposed_total,
            infected=infected_total,
            recovered=recovered_total,
            dead=dead_total,
            initial_population=total
        )

        return {'total':total, 'cities':cities}
    
    def set_action(self, act, city):
        """ Set the action variables in a given city
        
        
        Args:
            act (dict): a dict containing the following keys with boolean values
            confine (bool): should the city be confined?
            isolate (bool): should the city be isolated?
            vaccinate (bool): should the city be vaccinated?
            hospital (bool): should the city be given extra_hospital_beds?

        Returns:
            None
        """

        self.c_confined[city] = self.confinement_effectiveness if act['confinement'] else 1
        self.c_isolated[city] = self.isolation_effectiveness if act['isolation'] else 1
        self.extra_hospital_beds[city] = self.extra_hospital_effectiveness if act['hospital'] else 1
        self.vaccinate[city] = self.vaccination_effectiveness if act['vaccinate'] else 0

    def get_action(self,):
        _c = list(self.c_confined.keys())[0]
        return {'confinement': (self.c_confined[_c] != 1),
        'isolation': (self.c_isolated[_c] != 1),
        'vaccinate': (self.vaccinate[_c] != 0),
        'hospital': (self.extra_hospital_beds[_c] != 1),
        }
        
    
    
    def start_epidemic(self, seed=10, sources=1, prop=0.01):
        """ Starts the epidemic (infects a given proportion of the population in one or more randomly chosen cities).
        
        
        Args:
            seed (int): the random seed 
            sources (int): the number of cities we want the epidemic to start from
            prop (float): the propotion of the population we initialy infect in a given city

        Returns:
            None
        """
        rd.seed(seed)
        np.random.seed(seed)

        start_cities = rd.choices(self.cities, k=sources)
        for c in start_cities:
            self.map.nodes[c]['e'] += prop
            self.map.nodes[c]['s'] -= prop

    def step(self):
        """ Perform one environment step (a week of dynamical system simulation)

        Returns : 
            obs [dict] : a dict containing observation from this step
                    
                    
            the dict is structured as 
                total   :   list(dict(measurements)) (over 7 days)
                city    :   dict(list(dict(measurements))) (over 7 days, per city)
                pop    :    dict(int) initial per-city population (to allow normalization)

    """
        _total_history = []
        _city_history = []
        # step through a week of simulation to produce one environment step
        for i in range(self.env_step_length*self.srate):
            self.step_dyn()
            params = self.epidemic_parameters(day=i/self.srate)
            _total_history.append(params['total'])
            _city_history.append(params['cities'])

        # output observations
        _total = Observables(
            infected = [t.infected for t in _total_history][self.srate-1::self.srate],
            dead =  [t.dead for t in _total_history][self.srate-1::self.srate],
        )
        _city = {c:Observables(
                    infected=[t[c].infected for t in _city_history][self.srate-1::self.srate],
                    dead=[t[c].dead for t in _city_history][self.srate-1::self.srate],
                ) for c in self.cities}
            
        _pop = {c: self.map.nodes[c]['pop'] for c in self.cities}
        
        return Observation(
            pop=_pop,
            city=_city,
            total=_total,
        )

    """ Step forward in the epidemic dynamics
        
        Parameters : 
            None

        Returns : 
            None
    """

    def step_dyn(self):
        """Perform one dynamic simulation step.
        """
        ds = {}
        de = {}
        di = {}
        dr = {}
        dd = {}

        for c in self.cities:

            # query the variables from the graph
            s = self.map.nodes[c]['s']
            e = self.map.nodes[c]['e']
            i = self.map.nodes[c]['i']
            r = self.map.nodes[c]['r']
            d = self.map.nodes[c]['d']

            # compute the derivative terms

            # city - to city contagion
            stoch_t0 = np.max(
                [np.random.normal(self.tau_0, self.var_tau_0), 0])
            sum_term = self.c_isolated[c]*stoch_t0 * np.sum([self.map.nodes[a]['i']*self.map.edges[(
                a, c)]['tau']*self.c_isolated[a] for a in nx.neighbors(self.map, c)])

            # incidence rate
            stoch_alpha = np.max(
                [np.random.normal(self.alpha*self.c_confined[c], self.var_alpha), 0])
            new_exposed = stoch_alpha * (s * i + sum_term)

            # vaccination
            stoch_mu = self.vaccination_effectiveness / \
                self.map.nodes[c]['pop'] if self.vaccinate[c] else 0
            new_vaccinated = np.fmax(
                np.fmin(float(stoch_mu*s), float(stoch_mu)), 0)

            # exposure to infection flow
            stoch_eta = np.max([np.random.normal(self.eta, self.eta), 0])
            new_infected = stoch_eta * e

            # exposure to recovered flow
            stoch_beta = np.max(
                [np.random.normal(self.beta, self.var_beta), 0])
            new_recovered = stoch_beta * i

            # death rate
            stoch_zeta = np.max(
                [np.random.normal(self.zeta, self.var_zeta), 0])
            new_deaths = stoch_zeta * i * i * self.extra_hospital_beds[c]

            # loss of immunity rate
            stoch_gamma = np.max(
                [np.random.normal(self.gamma, self.var_gamma), 0])
            new_suceptible = stoch_gamma * r

            # compute the derivatives
            ds[c] = new_suceptible - new_exposed - new_vaccinated
            de[c] = new_exposed - new_infected
            di[c] = new_infected - new_recovered - new_deaths
            dr[c] = new_recovered - new_suceptible + new_vaccinated
            dd[c] = new_deaths

        for c in self.cities:
            # Euler integration step
            self.map.nodes[c]['s'] += ds[c]*self.dt
            self.map.nodes[c]['e'] += de[c]*self.dt
            self.map.nodes[c]['i'] += di[c]*self.dt
            self.map.nodes[c]['r'] += dr[c]*self.dt
            self.map.nodes[c]['d'] += dd[c]*self.dt
            self.map.nodes[c]['s'] = max(self.map.nodes[c]['s'],0)
            self.map.nodes[c]['e'] = max(self.map.nodes[c]['e'],0)
            self.map.nodes[c]['i'] = max(self.map.nodes[c]['i'],0)
            self.map.nodes[c]['r'] = max(self.map.nodes[c]['r'],0)
            self.map.nodes[c]['d'] = max(self.map.nodes[c]['d'],0)
