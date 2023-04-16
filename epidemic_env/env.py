"""Custom Environment that subclasses gym env."""

import gym
import torch
from gym.spaces import Space
import torch
from epidemic_env.dynamics import ModelDynamics, Parameters, Observables, Observation
from datetime import datetime as dt
from collections import namedtuple
from typing import Dict, List, Tuple, Any, Callable
from dataclasses import dataclass

ACTION_CONFINE = 1
ACTION_ISOLATE = 2
ACTION_HOSPITAL = 3
ACTION_VACCINATE = 4
SCALE = 100 #TODO : abstract that constant away

CONST_REWARD = 7
DEATH_COST = 1.3e4 
ANN_COST = 6
ISOL_COST = 1.5
CONF_COST = 6
VACC_COST = 0.08
HOSP_COST = 1

RewardTuple = namedtuple('RewardTuple',['reward','dead','conf','ann','vacc','hosp','isol'])


@dataclass
class  Log():
    """Contains a log of the sim parameters for the entire country on a given day d
    """
    total: Parameters
    city: Dict[str,Parameters]
    action: Dict[str,bool]
    
class Env(gym.Env):
    """Environment class, subclass of [gym.Env](https://www.gymlibrary.dev)."""
    metadata = {'render.modes': ['human']}
    
    def __init__(self,  dyn:ModelDynamics, 
                        action_space:Space=None,  # TODO : Replace with a fixed dict-space
                        observation_space:Space=None,
                        ep_len:int=30, 
                        action_preprocessor:Callable=lambda x,y:x, 
                        observation_preprocessor:Callable=lambda x,y:x, 
                        )->None:
        """**TODO describe:**
        
        Action Spaces (per mode)

        Modes 'binary', 'toggle, 'multi', 'factored' ==> TODO : Remove cases, use preprocessing functions
        
        pass -> preprocessor function
        
        pass -> output action space

        Args:
            dyn (ModelDynamics): Model Dynamics environment
            action_space (Space): action space
            observation_space (Space): observation space
            ep_len (int, optional): length of one episode. Defaults to 30.
            action_preprocessor (_type_, optional): preprocesses the actions. Defaults to lambdax:x.
            observation_preprocessor (_type_, optional): preprocesses the observations. Defaults to lambdax:x.
        """        
        super(Env, self).__init__()

        self.ep_len = ep_len
        self.dyn = dyn
    
        self.action_space = action_space
        self.observation_space = observation_space
            
        self.action_preprocessor        = action_preprocessor
        self.observation_preprocessor   = observation_preprocessor
        
        self.reward = torch.Tensor([0]).unsqueeze(0)
        self.reset()

    def compute_reward(self, obs:Observation)->RewardTuple:
        """Computes the reward \(R(s^{(t)},a^{(t)})\) from an observation dictionary `obs`:
        
        $$
            \\begin{aligned}
            \\textbf{Reward} &&
            R(s^{(t)},a^{(t)}) =  R_\\text{c}
            - \mathcal{C}(a^{(t)})
            - D \cdot \Delta d_\\text{city}^{(t)}\\\\
            \\textbf{Action cost} &&
            \mathcal{C}(a^{(t)}) =  
            \mathcal{A}(a^{(t)}) 
            + \mathbf{1}_{vac}  \cdot V
            + \mathbf{1}_{hosp} \cdot H
            + \mathbf{1}_{conf} \cdot C
            + \mathbf{1}_{isol} \cdot I \\\\
            \\textbf{Annoucement costs} &&
            \mathcal{A}(a^{(t)})  = A \cdot (\mathbf{1}^+_\\text{vac} + \mathbf{1}^+_\\text{hosp} + \mathbf{1}^+_\\text{conf})
        \end{aligned}
        $$

        Args:
            obs (Observation): The observation from the ModelDynamics class.

        Returns:
            RewardTuple: the reward and all of it's components
        """
        def compute_death_cost():
            dead = 0
            for city in self.dyn.cities:
                if len(obs.city[city].dead) > 1:
                    dead += DEATH_COST * \
                        (obs.city[city].dead[-1] - obs.city[city].dead[0]) / (self.dyn.total_pop) # Not great but close enough
                else:
                    dead += DEATH_COST * \
                        obs.city[city].dead[-1] / \
                        (self.dyn.total_pop)
            return dead
        def compute_isolation_cost():
            isol = 0
            for city in self.dyn.cities:
                isol += ISOL_COST * \
                    int(self.dyn.c_isolated[city] == self.dyn.isolation_effectiveness) * \
                    obs.pop[city] / (self.dyn.total_pop)
            return isol
        def compute_confinement_cost():
            conf = 0
            for city in self.dyn.cities:
                conf += CONF_COST * \
                    int(self.dyn.c_confined[city] == self.dyn.confinement_effectiveness) * \
                    obs.pop[city] / (self.dyn.total_pop)
            return conf
        def compute_annoucement_cost():
            announcement = 0
            if self._get_info().action['confinement'] and not self.last_info.action['confinement']:
                announcement += ANN_COST
            if self._get_info().action['isolation'] and not self.last_info.action['isolation']:
                announcement += ANN_COST
            if self._get_info().action['vaccinate'] and not self.last_info.action['vaccinate']:
                announcement += ANN_COST
            return announcement
        def compute_vaccination_cost():
            vacc = int(self.dyn.vaccinate['Lausanne'] != 0) * VACC_COST
            return vacc
        def compute_hospital_cost():
            hosp = (self.dyn.extra_hospital_beds['Lausanne'] != 1) * HOSP_COST
            return hosp

        dead = compute_death_cost()
        conf = compute_confinement_cost()
        ann = compute_annoucement_cost()
        vacc = compute_vaccination_cost()
        hosp = compute_hospital_cost()
        isol = compute_isolation_cost()

        rew = CONST_REWARD - dead - conf - ann - vacc - hosp
        return RewardTuple(torch.Tensor([rew]).unsqueeze(0), dead, conf, ann, vacc, hosp, isol)

    def get_obs(self, obs:Observation)->torch.Tensor:
        """Generates an observation tensor from a dictionary of observations.

        Args:
            obs (Observation): the observations dictionary.

        Raises:
            Exception: when the mode is incorrectly implemented.

        Returns:
            torch.Tensor: the observation tensor.
        """
        return self.observation_preprocessor(obs,self.dyn)
    
    def _parse_action(self, a):        
        return self.action_preprocessor(a,self.dyn)
        
    def _get_info(self)->Dict[str,Any]:
        """Grabs the dynamical system information dictionary from the simulator.

        Returns:
            Dict[str,Any]: The information dictionary.
        """
        _params = self.dyn.epidemic_parameters(self.day)
        return Log(
            total=_params['total'],
            city=_params['cities'],
            action={
                'confinement': (self.dyn.c_confined['Lausanne'] != 1),
                'isolation': (self.dyn.c_isolated['Lausanne'] != 1),
                'vaccinate': (self.dyn.vaccinate['Lausanne'] != 0),
                'hospital': (self.dyn.extra_hospital_beds['Lausanne'] != 1),
            },
        )

    def step(self, action:int)->Tuple[torch.Tensor,torch.Tensor,Observation]:
        """Perform one environment step.

        Args:
            action (int): the action

        Returns:
            Tuple[torch.Tensor,torch.Tensor,Dict[str,Any]]: A tuple containing
            - in element 1
        """
        self.day += self.dyn.env_step_length
        self.last_action = action
        self.last_info = self._get_info()
        for c in self.dyn.cities:
            self.dyn.set_action(self._parse_action(action), c)
        _obs = self.dyn.step()
        self.last_obs = self.get_obs(_obs)

        r = self.compute_reward(_obs)
        self.reward     = r.reward
        self.dead_cost  = r.dead
        self.conf_cost  = r.conf
        self.ann_cost   = r.ann
        self.vacc_cost  = r.vacc
        self.hosp_cost  = r.hosp
        self.isol       = r.isol

        done = self.day >= self.ep_len*self.dyn.env_step_length
        return self.last_obs, self.reward, done, self._get_info()

    def reset(self, seed:int=None)->Tuple[torch.Tensor,Dict[str,Any]]:
        """Reset the state of the environment to an initial state

        Args:
            seed (int, optional): random seed (for reproductability). Defaults to None.

        Returns:
            Tuple[torch.Tensor,Dict[str,Any]]: a tuple containing, in element 0 the observation tensor, in element 1 the information dictionary
        """
        self.last_action = 0
        self.day = 0
        self.dead_cost = 0
        self.conf_cost = 0
        self.ann_cost = 0
        self.vacc_cost = 0
        self.hosp_cost = 0
        self.isol = 0
        self.dyn.reset()
        if seed is None:
            self.dyn.start_epidemic(int(dt.now().second))
        else:
            self.dyn.start_epidemic(seed)

        _obs = self.dyn.step() # This obs is Obs class
        self.last_obs = self.get_obs(_obs) # This obs might be tensorized
        self.last_info = self._get_info()
        return self.last_obs, self.last_info

    def render(self, mode='human', close=False): 
        epidemic_dict = self.dyn.epidemic_parameters(self.day)
        print('Epidemic state : \n   - dead: {}\n   - infected: {}'.format(
            epidemic_dict['total']['dead'], epidemic_dict['total']['infected']))


