from abc import abstractmethod
import jax.numpy as jnp
from jax.lax import fori_loop
from jax import jit, vmap, value_and_grad, random, config
from jax.scipy.stats.multivariate_normal import logpdf
from jax.scipy.linalg import block_diag
import optax
from tqdm import tqdm
import numpy as np
from difai.aif_tools_jax import unscented, softmax_jax, kl_jax, kl_normal_normal, gen_fixed_plans, refactor_noise_params, load_yaml_file
from pathlib import Path

# Epsilon that is added to prevent numerical problems
EPS = 1e-12
DEFAULT_CONFIG_PATH = Path(__file__).parent / "default_config.yaml"
# Enable 64-bit precision in JAX (needed for the optimization)
config.update("jax_enable_x64", True)

class AIF_Env(object):
    ''' Interface for an environment of an active inference agent. This could be either a generative process or a generative model.'''
    def __init__(self, x0, dt, sys_params, add_params={}):
        self.x0 = x0 # Initial state
        self.dt = dt # Length of one time step
        self.sys_params = sys_params # System parameters
        self.non_negative_sys_params = [] # List of Non-negative system parameters (indices, e.g. [0, 1])
        self.dim_action = 0 # dimension of the action space
        self.dim_sys_params = len(self.sys_params) # dimension of the system parameters
        self.dim_observation = 0 # dimension of the observation space
        self._forward =  lambda x, u, random_realisation=None, key=None: self._forward_complete(x, u, self.dt, random_realisation, key, **self.sys_params)
        self.add_params = add_params # Dictionary with additional parameters
        self.jitable = True # Flag to indicate whether the environment is jitable 
        self.reset() # Reset the environment to the initial state

    def reset(self):
        """ Reset the environment to the initial state and re-jit the dynamics function """
        self.dim_sys_params = len(self.sys_params) 
        self.x = self.x0 
        self._forward =  lambda x, u, random_realisation=None, key=None: self._forward_complete(x, u, self.dt, random_realisation, key, **self.sys_params)
        self._get_observation =  lambda x: self._get_observation_complete(x, **self.sys_params)
        self.jitting()

    def jitting(self):
        """ Jit the dynamics function if possible. """
        if self.jitable:
            self.forward = jit(self._forward) # Jit the dynamics function
        else:
            self.forward = self._forward

    def step(self, u, random_realisation=None, key=None):
        """ Forward the generative process by one timestep with control u and updates the environment state.
        Args:
            u (array): Control input to the system.
            random_realisation (array): Random realisation for stochastic dynamics (optional).
            key (jax.random.PRNGKey): Random key for stochastic dynamics (optional).
        """
        self.x = self.forward(self.x, u, random_realisation, key) # Forward the environment by one timestep with control u

    @staticmethod
    @abstractmethod
    def _get_observation_complete(x, **sys_params):
        """ Get the (deterministic) observation from the state x given system parameters.
        Important: Result needs to be 1D-Array, even if the observation is a scalar.
        Args:
            x (array): State of the system.
            sys_params (dict/array): System parameters.
        Returns:
            o (array): Observation of the system.
        """
        raise NotImplementedError("_get_observation_complete method not implemented")

    @staticmethod
    @abstractmethod
    def _forward_complete(x, u, dt, random_realisation=None, key=None, **sys_params):
        """ Function to add probabilistic (discretized) system dynamics of the generative process
        Args:
            x (array): Initial state of the system.
            u (array): Control input to the system.
            dt (float): Length of one time step.
            random_realisation (array): Random realisation for stochastic dynamics (optional).
            key (jax.random.PRNGKey): Random key for stochastic dynamics (optional).
            sys_params (dict/array): System parameters.
        Returns:
            x (array): Resulting state of the system after applying the control input.
        """
        raise NotImplementedError("_forward_complete_probabilistic method not implemented")

# Active Inference Agent
class AIF_Agent:
    def __init__(self, generative_model, noise_params=None, params=None):            
        """Creates an Active Inference agent for the given environment. 
        Args:
            generative_model (AIF_Env): The generative model of the agent.
            noise_params (dict): Dictionary with noise parameters. Keys are the name of the noise source ('observation_std', 'state_dependent_obs_noise', 'constant_motor_noise', 'signal_dependent_noise'), and values are tuples containing the indices of the affected dimensions.
            params (dict): Dictionary with additional parameters for the agent. If None, default parameters are used.
        """   
        self.generative_model = generative_model # 
        if params is not None:
            self.params = params
        else:
            self.params = {}
            self.set_params_with_defaults()
        self.dt = generative_model.dt # length of one time step (discretization interval)
        self.params['dim_state'] = self.generative_model.x0.shape[0]  #self.generative_model.dim_state
        self.params['dim_observation'] = self.generative_model.dim_observation
        self.params['dim_dynamics'] = self.generative_model.dim_sys_params
        self.params['dim_action'] = self.generative_model.dim_action
        # Noise settings
        if noise_params is not None:
            noise_params, dim_noise = refactor_noise_params(noise_params)
        else:
            noise_params = {}
            dim_noise = 0
        self.params['dim_noise'] = dim_noise
        self.params['noise_params'] = noise_params  # Refactored Dictionary with noise parameters
        
        self.params['non_negative_sys_params'] = self.generative_model.non_negative_sys_params
        self.params.update(noise_params)  # Add noise parameters to the agent parameters to identify applied noise

        self._get_observation_complete = generative_model._get_observation_complete # function obtaining the deterministic observation, i.e., o = g(s)

        self.current_plan = None
        self.cur_step = 0
        
        # Hierarchical AIF
        self.belief_state_history = []
        self.action_history = []
        self.observation_history = []
    
    def set_params(self, 
               a_lims=None, n_plans=None, horizon=None, multistep=None, n_samples_o=None, 
               n_steps_o=None, lr_o=None, use_info_gain=None, use_observation_preference=None, 
               use_pragmatic_value=None, scale_pragmatic_value=None, select_max_pi=None, n_samples_ig_s=None, n_samples_ig_o=None, 
               n_samples_obs_pref_s=None, n_samples_obs_pref_o=None, initial_belief_state=None, 
               initial_belief_sys=None, initial_belief_noise=None, n_samples_a=None, 
               n_samples_a_combine=None, n_samples_a_noise_sys=None, use_complete_ukf=None, 
               exp_normal_sys_params=None, C_index=None, sys_dependent_C=None, state_dependent_C=None, 
               action_prior=None, use_fixed_plans=None, state_scaling=None, standardise_state=None, reaction_time=None):
        """
        Sets the parameters of the agent.
        Args:
            a_lims (list of np arrays): Action limits. First entry is the lower bound, second entry is the upper bound.
            n_plans (int): Number of plans rolled out during action selection. Default is 100.
            horizon (int): Planning horizon. Default is 1.
            multistep (int): Control update frequency (same control is applied for multistep environment steps). Default is 1.
            n_samples_o (int): Observation samples for belief update. Default is 100.
            n_steps_o (int): Optimization steps after new observation. Default is 50.
            lr_o (float/tuple): Learning rate of optimization after new observation. Default is 1e-3.
            use_info_gain (bool): Score actions by information gain. Default is True.
            use_observation_preference (bool): Score actions by observation preference. Default is False.
            use_pragmatic_value (bool): Score actions by pragmatic value. Default is True.
            scale_pragmatic_value (float): Scale the pragmatic value. Default is 1.
            select_max_pi (bool): Sample plan (False) or select max negative Expected Free Energy (True). Default is True.
            n_samples_ig_s (int): State samples for information gain. Default is 3.
            n_samples_ig_o (int): Observation samples for information gain. Default is 3.
            n_samples_obs_pref_s (int): State samples for observation preference. Default is 10.
            n_samples_obs_pref_o (int): Observation samples for observation preference. Default is 5.
            initial_belief_state (list): Initial belief about the state defined as a list of two arrays [mean, cov]. Default is None.
            initial_belief_sys (list): Initial belief about the system parameters defined as a list of two arrays [mean, cov]. Default is None.
            initial_belief_action (list): Initial belief about the action defined as a list of two arrays [mean, cov]. Default is None.
            use_observation_belief (bool): If True, the agent builds a belief over the system's observation noise standard deviation; If False, only build a belief over the system's state. Default is True.
            n_samples_a (int): Number of state samples for action update. Default is 10.
            n_samples_a_combine (int): Number of samples used to combine state beliefs if UKF is not used. Default is 200.
            n_samples_a_noise_sys (int): Number of samples used to estimate noise system parameters. Default is 10.
            use_complete_ukf (bool): Use the unscented Kalman filter for both state and system parameters in action update. Default is False.
            C_index (int): Index of the observation parameter that is used to calculate the pragmatic value. Default is 0.
            sys_dependent_C (tuple): If not None, the mean of the preference distribution is dependent on the system parameters. First entry is the index of the observation/state vector that is affected, second entry is the index of the system parameter that defines the mean of the preference distribution.
            state_dependent_C (tuple): If not None, the mean of the preference distribution is dependent on the state. First entry is the index of the observation/state vector that is affected, second entry is the index of the state that defines the mean of the preference distribution.
            use_state_based_action_prior (bool): If True, use state-based action prior. Default is False.
            action_prior (list): Prior for sampling possible actions defined as a list of two arrays [mean, cov]. Default is None.
            use_fixed_plans (bool): If True, use fixed plans for action selection. Default is False.
            state_scaling (list): Scaling factors for each state dimension. Default is None.
            standardise_state (bool): If True, standardise the state by dividing it by the scaling factors. Default is False.
        """
        # Add non-None parameters to self.params
        for key, value in locals().items():
            if key != "self" and value is not None:
                self.params[key] = value

    def set_params_with_defaults(self, a_lims=None, n_plans=None, horizon=None, multistep=None, n_samples_o=None, 
               n_steps_o=None, lr_o=None, use_info_gain=None, use_observation_preference=None, 
               use_pragmatic_value=None, scale_pragmatic_value=None, select_max_pi=None, n_samples_ig_s=None, n_samples_ig_o=None, 
               n_samples_obs_pref_s=None, n_samples_obs_pref_o=None, initial_belief_state=None, 
               initial_belief_sys=None, initial_belief_noise=None, n_samples_a=None, 
               n_samples_a_combine=None, n_samples_a_noise_sys=None, use_complete_ukf=None, 
               exp_normal_sys_params=None, C_index=None, sys_dependent_C=None, state_dependent_C=None, 
               action_prior=None, use_fixed_plans=None, state_scaling=None, standardise_state=None, reaction_time=None):
        """
        Sets the parameters of the agent.
        Args:
            a_lims (list of np arrays): Action limits. First entry is the lower bound, second entry is the upper bound.
            n_plans (int): Number of plans rolled out during action selection. Default is 100.
            horizon (int): Planning horizon. Default is 1.
            multistep (int): Control update frequency (same control is applied for multistep environment steps). Default is 1.
            n_samples_o (int): Observation samples for belief update. Default is 100.
            n_steps_o (int): Optimization steps after new observation. Default is 50.
            lr_o (float/tuple): Learning rate of optimization after new observation. Default is 1e-3.
            use_info_gain (bool): Score actions by information gain. Default is True.
            use_observation_preference (bool): Score actions by observation preference. Default is False.
            use_pragmatic_value (bool): Score actions by pragmatic value. Default is True.
            scale_pragmatic_value (float): Scale the pragmatic value. Default is 1.
            select_max_pi (bool): Sample plan (False) or select max negative Expected Free Energy (True). Default is True.
            n_samples_ig_s (int): State samples for information gain. Default is 3.
            n_samples_ig_o (int): Observation samples for information gain. Default is 3.
            n_samples_obs_pref_s (int): State samples for observation preference. Default is 10.
            n_samples_obs_pref_o (int): Observation samples for observation preference. Default is 5.
            initial_belief_state (list): Initial belief about the state defined as a list of two arrays [mean, cov]. Default is None.
            initial_belief_sys (list): Initial belief about the system parameters defined as a list of two arrays [mean, cov]. Default is None.
            initial_belief_action (list): Initial belief about the action defined as a list of two arrays [mean, cov]. Default is None.
            use_observation_belief (bool): If True, the agent builds a belief over the system's observation noise standard deviation; If False, only build a belief over the system's state. Default is True.
            n_samples_a (int): Number of state samples for action update. Default is 10.
            n_samples_a_combine (int): Number of samples used to combine state beliefs if UKF is not used. Default is 200.
            n_samples_a_noise_sys (int): Number of samples used to estimate noise system parameters. Default is 10.
            use_complete_ukf (bool): Use the unscented Kalman filter for both state and system parameters in action update. Default is False.
            C_index (int): Index of the observation parameter that is used to calculate the pragmatic value. Default is 0.
            sys_dependent_C (tuple): If not None, the mean of the preference distribution is dependent on the system parameters. First entry is the index of the observation/state vector that is affected, second entry is the index of the system parameter that defines the mean of the preference distribution.
            state_dependent_C (tuple): If not None, the mean of the preference distribution is dependent on the state. First entry is the index of the observation/state vector that is affected, second entry is the index of the state that defines the mean of the preference distribution.
            use_state_based_action_prior (bool): If True, use state-based action prior. Default is False.
            action_prior (list): Prior for sampling possible actions defined as a list of two arrays [mean, cov]. Default is None.
            use_fixed_plans (bool): If True, use fixed plans for action selection. Default is False.
            state_scaling (list): Scaling factors for each state dimension. Default is None.
            standardise_state (bool): If True, standardise the state by dividing it by the scaling factors. Default is False.
        """
        self.params.update(load_yaml_file(DEFAULT_CONFIG_PATH))  # Load default parameters
        changed_params = locals()
        changed_params.pop('self', None)  # Remove self from the parameters
        self.set_params(**changed_params)


    def load_parameters_with_defaults(self, user_config_path):
        """
        Load user and default parameters from YAML files and merge them.

        Args:
            user_config_path (str): Path to the user-specified config YAML.
            default_config_path (str): Path to the default config YAML.

        """
        default_params = load_yaml_file(DEFAULT_CONFIG_PATH)
        user_params = load_yaml_file(user_config_path)

        # Merge: user overrides defaults
        combined_params = default_params.copy()
        combined_params.update(user_params)

        self.params.update(combined_params)

    def set_initial_beliefs(self, initial_belief_state=None, initial_belief_sys=None, initial_belief_noise=None):
        """
        Set the initial beliefs for the agent.

        Args:
            initial_belief_state (list): Initial belief about the state defined as a list of two arrays [mean, cov].
            initial_belief_sys (list): Initial belief about the system parameters defined as a list of two arrays [mean, cov].
            initial_belief_noise (list): Initial belief about the noise defined as a list of two arrays [mean, cov].
        """
        self.params["initial_belief_state"] = initial_belief_state 
        self.params["initial_belief_sys"] = initial_belief_sys 
        self.params["initial_belief_noise"] = initial_belief_noise 

    def set_preference_distribution(self, C, C_index, sys_dependent_C=None, state_dependent_C=None, use_observation_preference=True):
        """
        Set the preference distribution for the agent.
        Args:
            C (tuple): Preference distribution parameters. First entry is the mean, second entry is the covariance matrix.
            C_index (int): Index of the observation/state parameter that is used to calculate the pragmatic value.
            sys_dependent_C (tuple): If not None, the mean of the preference distribution is dependent on the system parameters. First entry is the index of the observation/state parameter, second entry is the index of the system parameter.
            state_dependent_C (tuple): If not None, the mean of the preference distribution is dependent on the state. First entry is the index of the observation/state parameter, second entry is the index of the state.
            use_observation_preference (bool): If True, the agent uses observation preference. Default is True.
        """
        self.params["use_observation_preference"] = use_observation_preference
        self.params["C_index"] = C_index
        self.params["sys_dependent_C"] = sys_dependent_C
        self.params["state_dependent_C"] = state_dependent_C
        self.C = C
        # Make sure C[1] is a 2D array (covariance matrix)
        self.C[1] = jnp.atleast_2d(self.C[1])
        self.C_entropy = 0.5 * jnp.log(jnp.linalg.det(self.C[1])) + self.C[0].shape[0]/2 +  self.C[0].shape[0]/2 * jnp.log(2 * jnp.pi)

    def initialize(self):
        self.params["has_observation_noise"] = "observation_std" in self.params
        
        if self.params["has_observation_noise"]:
            self.params["observation_noise_indices"] = self.params["observation_std"][0]
        else:
            self.params["observation_noise_indices"] = []

        if self.params["initial_belief_state"] is None:
            self.params["initial_belief_state"] = [jnp.zeros((self.params['dim_state'],)), jnp.eye(self.params['dim_state'])]
        if self.params["initial_belief_sys"] is None:
            self.params["initial_belief_sys"] = [jnp.zeros((self.params['dim_dynamics'],)), jnp.eye(self.params['dim_dynamics'])]    
        if self.params["initial_belief_noise"] is None:
            self.params["initial_belief_noise"] = [jnp.zeros((self.params['dim_noise'],)), jnp.eye(self.params['dim_noise'])]
 
        if self.params["use_fixed_plans"]:
            self.params["fixed_plans"] = gen_fixed_plans(self.params["a_lims"], self.params["horizon"], self.params["dim_action"], self.params["n_plans"], uniform=(self.params["use_fixed_plans"]=='uniform'))
            self.params['n_plans'] = self.params["fixed_plans"].shape[0]

        # Make sure everything is at least 1d/2d
        self.params["initial_belief_state"][0] = jnp.atleast_1d(self.params["initial_belief_state"][0])
        self.params["initial_belief_state"][1] = jnp.atleast_2d(self.params["initial_belief_state"][1])
        self.params["initial_belief_sys"][0] = jnp.atleast_1d(self.params["initial_belief_sys"][0])
        self.params["initial_belief_sys"][1] = jnp.atleast_2d(self.params["initial_belief_sys"][1])
        self.params["initial_belief_noise"][0] = jnp.atleast_1d(self.params["initial_belief_noise"][0])
        self.params["initial_belief_noise"][1] = jnp.atleast_2d(self.params["initial_belief_noise"][1])

        self.reset()   
        
        # Check consistency in params
        assert jnp.all(jnp.linalg.eigh(self.params["initial_belief_state"][1])[0] > 0), "Initial belief covariance about states and actions needs to be positive definite."
        if self.params['dim_dynamics'] > 0:
            assert jnp.all(jnp.linalg.eigh(self.params["initial_belief_noise"][1])[0] > 0), "Initial belief covariance about noise needs to be positive definite."
        if self.params['dim_noise'] > 0:
            assert jnp.all(jnp.linalg.eigh(self.params["initial_belief_sys"][1])[0] > 0), "Initial belief covariance about sys paramsneeds to be positive definite."

        #Jitting
    def _jit_methods(self):

        ## Former Markov Blanket methods
        _lambda_apply_control_noise_given_d = lambda a, d: self._apply_control_noise_given_d(a, d, self.params)
        self.apply_control_noise =  jit(lambda a, key: self._apply_control_noise(a, _lambda_apply_control_noise_given_d,  self.params, key))

        self.sample_o_given_s_d_sys = jit(lambda s, d, sys: self._sample_o_given_s_d_sys(s, d, sys, self._get_observation_complete, self.params))

        _lambda_sample_s1_given_s_a_d_sys = lambda s, a, d, sys: self._sample_s1_given_s_a_d_sys(s, a, d, self.dt, sys, self.generative_model._forward_complete, _lambda_apply_control_noise_given_d, self.params)
        self.sample_s1_given_s_a_d_sys = jit(_lambda_sample_s1_given_s_a_d_sys)


        _lambda_update_belief_state = lambda belief_state, belief_noise, belief_sys, a, key: self._update_belief_state(belief_state, belief_noise, belief_sys, a, self.sample_s1_given_s_a_d_sys, self.params, key)
        self.update_belief_state = jit(_lambda_update_belief_state)

        _lambda_sample_obs_noise_params = lambda belief_noise, n_samples, key: self._sample_obs_noise_params(belief_noise, n_samples, self.params, key)
        self.sample_obs_noise_params = jit(_lambda_sample_obs_noise_params)
        _lambda_update_belief_state_obs = lambda belief_state, belief_noise, belief_sys, o, key: self._update_belief_state_obs(belief_state, belief_noise, belief_sys, o, self._get_observation_complete, _lambda_sample_obs_noise_params, self.params, key)
        self.update_belief_state_obs = jit(_lambda_update_belief_state_obs)

        self.select_action = jit(lambda belief_state, belief_noise, belief_sys, key=None:
                                self._select_action(belief_state, belief_noise, belief_sys, self.C, _lambda_update_belief_state_obs, _lambda_update_belief_state, self._get_observation_complete, self._calc_pragmatic_value, _lambda_sample_obs_noise_params, self.params, key))
    def reset(self):
        # initialize belief
        self.belief_state = self.params["initial_belief_state"]
        self.belief_sys = self.params["initial_belief_sys"]
        self.belief_noise = self.params["initial_belief_noise"]
        # Jit updates, select action and step methods
        self._jit_methods()

    @staticmethod
    def _apply_control_noise_given_d(a, d, params):
        ''' Function to apply control noise to the action a
        Args:
            a: action
            d: noise realisation
            params: Parameters of the agent
            dim_action: number of actions
        Returns:
            a: action with added noise
        '''
        dim_action = params['dim_action']
        if 'signal_dependent_noise' in params:
            noise_indices = params['signal_dependent_noise'][0]
            action_indices = params['signal_dependent_noise'][1]
            a += jnp.zeros((dim_action,)).at[action_indices].set(jnp.multiply((a if dim_action == 1 else a[action_indices]), d[noise_indices]))
            a = a[0] if dim_action == 1 else a

        if 'constant_motor_noise' in params:
            noise_indices = params['constant_motor_noise'][0]
            action_indices = params['constant_motor_noise'][1]
            a +=  jnp.zeros((dim_action,)).at[action_indices].set(d[noise_indices])
            a = a[0] if dim_action == 1 else a
        return a
    
    @staticmethod
    def _apply_control_noise(a, _apply_control_noise_given_d, params, key):
        ''' Function to apply control noise to the action a
        Args:
            a: action
            noise_params: dictionary with noise parameters
            dim_action: number of actions
            key: unused jax random key
        Returns:
            a: action with added noise
        '''
        if 'signal_dependent_noise' in params:
            noise_indices = params['signal_dependent_noise'][0]
            key, use_key = random.split(key)
            d1 = random.multivariate_normal(use_key, jnp.zeros((noise_indices.shape[0],)), jnp.diag(jnp.square(params['signal_dependent_noise'][2])))
        else:
            d1 = jnp.zeros((0,))
        if 'constant_motor_noise' in params:
            noise_indices = params['constant_motor_noise'][0]
            key, use_key = random.split(key)
            d2 = random.multivariate_normal(use_key, jnp.zeros((noise_indices.shape[0],)), jnp.diag(jnp.square(params['constant_motor_noise'][2])))
        else:
            d2 = jnp.zeros((0,))
        d = jnp.concatenate((d1, d2), axis=0)  # Concatenate noise vectors
        a = _apply_control_noise_given_d(a, d)  # Apply control noise to the action
        return a

  
    @staticmethod
    def _sample_s1_given_s_a_d_sys(s, a, d, dt, sys, dynamics_total, _apply_control_noise_given_d, params):
        ''' Function to sample the next state s1 given the current state s and action a with complete definition of system dynamics through variables
        Args:
            s: current state
            a: action
            d: noise realisation
            dt: time step duration
            sys: variables defining the system dynamics
            dynamics_total: function defining the (discretized) system dynamics with complete definition of system dynamics through variables, i.e., s1 = f(s, a, dt, *sys)
            _apply_control_noise_given_d: function to apply control noise to the action a
            params: Parameters of the agent
            key: unused jax random key
        Returns:
            s1: sampled next state
            a: aplied action with noise
        '''
        a = _apply_control_noise_given_d(a, d)
        noise_realisation = None
        return dynamics_total(s, a, dt, noise_realisation, None, *sys), a #    

    @staticmethod
    def _sample_o_given_s_d_sys(s, d, sys, get_observation_complete, params): 
        ''' Function to sample the observation o given the state s
        Args:
            s: current state
            d: noise realisation
            sys: variables defining the system dynamics
            get_observation_complete: function defining the deterministic observation given state and system parameters, i.e., o = g(s, sys)
            params: Parameters of the agent
        Returns:
            o: sampled observation
        '''
        # Generate observation from any given state s
        dim_observation = params['dim_observation']
        obs = get_observation_complete(s, sys)
        if 'observation_std' in params:
            noise_indices = params['observation_std'][0]
            obs_indices = params['observation_std'][1]
            obs += jnp.zeros(dim_observation).at[obs_indices].set(d[noise_indices])
        if 'state_dependent_obs_noise' in params:
            noise_indices = params['state_dependent_obs_noise'][0]
            obs_indices = params['state_dependent_obs_noise'][1]
            state_indices = params['state_dependent_obs_noise'][2]
            obs += jnp.zeros(dim_observation).at[obs_indices].set(jnp.multiply((EPS+s[state_indices]), d[noise_indices]))
        return obs 
####
       
    @staticmethod
    def _sample_obs_noise_params(belief_noise, n_obs_samples, params, key):
        has_observation_noise = params["has_observation_noise"]
        dim_noise = params['dim_noise']
        dim_observation = params['dim_observation']
        obs_std = jnp.zeros((n_obs_samples, dim_observation))
        obs_var = jnp.zeros((n_obs_samples, dim_observation))
        if has_observation_noise:
            observation_noise_indices = params["observation_noise_indices"]
            noise_obs_noise_indices = params["observation_std"][0]
            
            # Sample observation noise
            key, use_key = random.split(key)
            if dim_noise > 1:
                noise_samples_log = random.multivariate_normal(use_key, belief_noise[0], belief_noise[1], shape=(n_obs_samples,))
            else:
                noise_samples_log = belief_noise[0] + jnp.sqrt(belief_noise[1]) * random.normal(use_key, shape=(n_obs_samples,))
                noise_samples_log = noise_samples_log.reshape((n_obs_samples,-1))
            noise_samples =  jnp.exp(noise_samples_log)[:, noise_obs_noise_indices]
            obs_std = obs_std.at[:, observation_noise_indices].set(noise_samples)
            obs_var = obs_var.at[:, observation_noise_indices].set(jnp.square(noise_samples))
        return obs_std, obs_var

    @staticmethod
    def _update_belief_state(belief_state, belief_noise, belief_sys, a, sample_s1_given_s_a_d_sys, params, key=None):
        """ Update belief over state given performed action and made observation (after action) 
                Args:
                    belief_state (list): Belief about the state [mean, cov]
                    belief_noise (list): Belief about the noise parameters [mean, cov]
                    belief_sys (list): Belief about the system parameters [mean, cov]
                    a: action
                    sample_s1_given_s_a_d_sys (function): Function to sample the next state given the current state and action, with complete definition of system dynamics through variables and noise realisation
                    params (dict): Parameters of the agent
                    key (jax.random.PRNGKey): Unused Jax Random key
                Returns:
                    belief_state (list): Updated belief about the state [mean, cov]
                    belief_states (list): List of beliefs about the state for each sample (only if noise is present)
        """
        dim_state = params['dim_state']
        dim_noise = params['dim_noise']
        dim_belief = len(belief_state[0])

        n_samples_a = params['n_samples_a']
        n_samples_a_combine = params['n_samples_a_combine']
        n_samples_a_noise_sys = params['n_samples_a_noise_sys']
        use_complete_ukf = params['use_complete_ukf']
        has_motor_noise = 'signal_dependent_noise' in params or 'constant_motor_noise' in params
        exp_normal_sys_params = params['exp_normal_sys_params']

        if not has_motor_noise:
            # System dynamics
            def sysfn(ssys):
                s = ssys[:dim_state]
                sys = ssys[dim_state:]
                if exp_normal_sys_params:
                    sys = jnp.exp(sys)

                s_new, _ = sample_s1_given_s_a_d_sys(s,a,None,sys)
                return jnp.hstack([s_new, sys])

            # State and dyn combined mean and cov of prior
            mean = jnp.hstack([belief_state[0], belief_sys[0]])
            cov = block_diag(belief_state[1], belief_sys[1]) 

            # Forward belief using unscented Kalman Filter
            ukf_mean, ukf_cov = unscented(mean, cov, fn = sysfn)  

            # Update belief about system state (retain belief about other parameters)
            belief_state = [ukf_mean[:dim_state], ukf_cov[:dim_state,:dim_state]]

            # No multiple updates
            belief_states = None
        else:
            # Sample noise parameters
            key, use_key = random.split(key)
            noise_vec_samples = random.multivariate_normal(use_key, *belief_noise, shape=(n_samples_a_noise_sys,))
            noise_vec_samples = jnp.exp(noise_vec_samples)

            # Sample system parameters
            key, use_key = random.split(key)
            sys_params_samples = random.multivariate_normal(use_key, *belief_sys, shape=(n_samples_a_noise_sys,))
            if exp_normal_sys_params:
                sys_params_samples = jnp.exp(sys_params_samples)
            
            def update_single(noise_vec, sys_params, key):
                # Dynamics for UKF
                def sysfn(sd):
                    s = sd[:dim_state] # state
                    d = sd[dim_state:] # Noise realisation
                    s_new, _ = sample_s1_given_s_a_d_sys(s, a, d, sys_params)
                    return jnp.hstack([s_new, d])

                # State and action and noise combined mean and cov of prior
                mean = jnp.hstack([belief_state[0], jnp.zeros((dim_noise,))])
                cov = block_diag(belief_state[1], jnp.diag(jnp.square(noise_vec))) 

                if use_complete_ukf:
                    ## Run the Unscented Kalman Filter
                    ukf_mean, ukf_cov = unscented(mean, cov, fn = sysfn)  
                    ##
                    belief_state_new = [ukf_mean[:dim_state], ukf_cov[:dim_state,:dim_state]]

                else:
                    # Sample from joint distribution of state and noise
                    key, use_key = random.split(key)
                    ssys = random.multivariate_normal(use_key, mean, cov, shape=(n_samples_a,))

                    # Apply dynamics
                    ss1 = jnp.apply_along_axis(sysfn, 1, ssys)[:,:dim_state] # Sampled states
                    # New mean
                    loc_new = jnp.nanmean(ss1, axis=0) #jnp.mean(ss1, axis=0)

                    # New cov (calculate cov from mean)
                    cov_new = (ss1-loc_new).T @ (ss1-loc_new) / (n_samples_a-1)

                    # Add small value to cov
                    cov_new += jnp.diag(jnp.ones(dim_belief)*EPS)

                    # Update belief about system state 
                    belief_state_new = [loc_new[:dim_state], cov_new[:dim_state,:dim_state]]

                return belief_state_new

            # Run update for each sample
            keys = random.split(key, n_samples_a_noise_sys)
            belief_states = vmap(update_single, in_axes= (0,0,0), out_axes=0)(noise_vec_samples, sys_params_samples, keys)

            ## Combine results
            # Choose n_samples_a_combine samples from the posterior beliefs
            key, use_key = random.split(key)
            chosen_indices = random.choice(use_key, n_samples_a_noise_sys, shape=(n_samples_a_combine,))

            # Sample states from a normal distributions in vmap
            batch_keys = random.split(key, n_samples_a_combine+1)
            key = batch_keys[0]
            use_keys = batch_keys[1:]
            sample_states = vmap(lambda id, k: random.multivariate_normal(k, belief_states[0][id], belief_states[1][id]), in_axes=(0,0))(chosen_indices, use_keys)

            # Calculate new mean and covariance
            new_mean = jnp.nanmean(sample_states, axis=0)
            new_cov = (jnp.nan_to_num(sample_states)-new_mean).T @ (jnp.nan_to_num(sample_states)-new_mean) / ((n_samples_a_combine-jnp.isnan(sample_states).any(axis=1).sum())-1)
            belief_state = [new_mean, new_cov]

        # Add small value to diagonal to ensure spd
        ukf_cov = belief_state[1]
        ukf_cov += jnp.diag(jnp.ones(dim_belief)*EPS)
        belief_state = [belief_state[0], ukf_cov]

        return belief_state, belief_states
    

    

    @staticmethod
    def _update_belief_state_obs(belief_state, belief_noise, belief_sys, o, get_observation_complete, sample_obs_noise_params, params, key):
        '''Performs Bayesian variational inference to update a prior belief when observing o.
        Args:
            belief_state (list): Belief about the state and action (if used) [mean, cov]
            belief_noise (list): Belief about the noise parameters [mean, cov]
            belief_sys (list): Belief about the system parameters [mean, cov]
            o (array): Observation
            get_observation_complete (function): Function to get the (deterministic) observation given the state and system parameters
            sample_obs_noise_params (function): Function to sample the observation noise parameters
            params (dict): Parameters of the agent
            key (jax.random.PRNGKey): Unused Jax Random key
        '''
        n_samples_o =  params["n_samples_o"]
        n_steps_o =  params["n_steps_o"]
        lr_o =  params["lr_o"]
        dim_state = params['dim_state']
        dim_belief = len(belief_state[0]) #params['dim_belief']
        dim_observation = params['dim_observation']
        tril_indices =  jnp.tril_indices(dim_belief) #params['tril_indices'] 
        prior_mean = belief_state[0]
        prior_cov = belief_state[1]

        # Normalise state
        if params['standardise_state']:
            # Calculate scaling factors for each state dimension
            # jaxprint("Prior before scaling {x}. cov {y}", x=prior_mean, y=prior_cov)
            state_scaling = jnp.array(params['state_scaling'])
            prior_mean = prior_mean / state_scaling
            cov_scaling = (state_scaling[:, None] * state_scaling[None, :])
            prior_cov = prior_cov * cov_scaling**(-1)

            # jaxprint("Prior after scaling {x}. cov {y}", x=prior_mean, y=prior_cov)
        # initialise belief to belief before observing
        opt_params = jnp.hstack([prior_mean, jnp.linalg.cholesky(prior_cov)[tril_indices].reshape((-1,))])

        # Sample system parameters
        key, use_key = random.split(key)
        sample_sys = random.multivariate_normal(use_key, belief_sys[0], belief_sys[1], shape=(n_samples_o,))

        # Sample noise belief
        key, use_key = random.split(key)
        _, sample_observation_var =  sample_obs_noise_params(belief_noise, n_samples_o, use_key)
        
        optimizer = optax.rmsprop(learning_rate=lr_o, eps=EPS)
        # optimizer = optax.adam(learning_rate=lr_o, eps=EPS)

        opt_state = optimizer.init(opt_params)
    
        def lossfct(opt_params, step_i, key):
            posterior_mean = opt_params[:dim_belief]
            # putting back together tril
            posterior_scale_tril = jnp.zeros((dim_belief,dim_belief)).at[tril_indices].set(opt_params[dim_belief:])
            posterior_cov = posterior_scale_tril @ posterior_scale_tril.T

            if params['standardise_state']:
            # Rescale posterior mean and covariance
                posterior_mean = posterior_mean * state_scaling
                posterior_cov = posterior_cov * cov_scaling
                # posterior_cov = posterior_cov * (state_scaling[:, None] * state_scaling[None, :])

            # Sample states 
            key, use_key = random.split(key)
            sample_states = random.multivariate_normal(use_key, posterior_mean, posterior_cov, shape=(n_samples_o,) )

            # Get observations from these states
            sample_observations = jnp.apply_along_axis(lambda xsys: get_observation_complete(xsys[:dim_state], *xsys[dim_state:]), 1, jnp.hstack([sample_states[:,:dim_state], sample_sys])) # sampled observations

            def nll_single(o_var):
                s_o = o_var[:dim_observation]
                s_o_var = o_var[dim_observation:]
                diff = s_o-o
                return 0.5 * (jnp.log(jnp.prod(s_o_var)) + jnp.dot(diff.T, (1/jnp.clip(s_o_var, min=EPS)) * diff) + s_o.shape[0] *  jnp.log(2 * jnp.pi))

            nll = jnp.apply_along_axis(nll_single, 1, jnp.hstack([sample_observations.reshape((-1,dim_observation)), sample_observation_var.reshape((-1,dim_observation))]))
            kl = kl_jax(posterior_mean, posterior_cov, prior_mean, prior_cov)

            loss = kl + nll.mean()

            return loss

        def train_step(step_i, opt_state, opt_params, key):
            loss, grads = value_and_grad(lossfct, argnums=0)(opt_params, step_i, key)

            ### optax
            updates, opt_state = optimizer.update(grads, opt_state)
            opt_params = optax.apply_updates(opt_params, updates)

            return loss, opt_state, opt_params

        def body_fun(i, carry):
            key, use_key = random.split(carry[0])
            opt_state = carry[2]
            opt_params = carry[3]
            loss, opt_state, opt_params = train_step(i, opt_state, opt_params, use_key)
            return (key, carry[1].at[i].set(loss), opt_state, opt_params)

        opt_result = fori_loop(0, n_steps_o, body_fun, (key, jnp.zeros((n_steps_o,)), opt_state, opt_params))

        ll = opt_result[1]
        opt_params = opt_result[3]

        posterior_mean = opt_params[:dim_belief]
        # putting back together tril
        posterior_scale_tril = jnp.zeros((dim_belief,dim_belief)).at[jnp.tril_indices(dim_belief)].set(opt_params[dim_belief:]) 

        posterior_cov = posterior_scale_tril @ posterior_scale_tril.T

        if params['standardise_state']:
            # Rescale posterior mean and covariance
            posterior_mean = posterior_mean * state_scaling
            posterior_cov = posterior_cov * cov_scaling
            # posterior_cov = posterior_cov * (state_scaling[:, None] * state_scaling[None, :])

        posterior_cov += jnp.diag(jnp.ones(dim_belief)*EPS) # Add small value to diagonal to ensure positive definiteness

        belief_state = [posterior_mean, posterior_cov]
        return belief_state, ll, lr_o
    
    @staticmethod
    def _calc_pragmatic_value(belief_state, C, params):
        C_index = params['C_index']
        marginal_theta = [belief_state[0][C_index], jnp.sqrt(belief_state[1][C_index,C_index])]
        dist = kl_normal_normal(*marginal_theta, *C)
        return -dist

    @staticmethod
    def _select_action(belief_state_prior, belief_noise_prior, belief_sys_params_prior, C, _update_belief, _update_belief_a, get_observation_complete, _calc_pragmatic_value, sample_obs_noise_params, params, key): # return plans, p of selecting each, and marginal p of actions
        a_lims = params['a_lims']
        n_plans = params['n_plans']
        horizon = params['horizon']
        multistep = params['multistep']
        select_max_pi = params['select_max_pi']
        scale_pragmatic_value = params['scale_pragmatic_value']
        n_samples_ig_s = params['n_samples_ig_s']
        n_samples_ig_o = params['n_samples_ig_o']
        n_samples_obs_pref_s = params['n_samples_obs_pref_s']
        n_samples_obs_pref_o = params['n_samples_obs_pref_o']
        n_samples_ig = n_samples_ig_s * n_samples_ig_o
        dim_state = params["dim_state"]
        dim_action = params["dim_action"]
        dim_observation = params["dim_observation"]
        has_observation_noise = params['has_observation_noise']
        use_info_gain = params['use_info_gain']
        use_observation_preference = params['use_observation_preference']
        action_prior = params['action_prior']
        use_fixed_plans = params['use_fixed_plans']
        sys_dependent_C = params['sys_dependent_C']
        state_dependent_C = params['state_dependent_C']
        C_index = params['C_index']
        
        if action_prior is None:
            if use_fixed_plans:
                plans = params["fixed_plans"]
            else:
                # Sample plans (actions) from uniform distribution based on action limits
                key, use_key = random.split(key)
                plans = random.uniform(key=use_key, shape=(n_plans, horizon, dim_action), minval=a_lims[0], maxval=a_lims[1])
        else:
            # Sample plans from action prior
            key, use_key = random.split(key)
            plans = random.multivariate_normal(key=use_key, mean=action_prior[0], cov=action_prior[1], shape=(n_plans, horizon,))

        if sys_dependent_C is not None or state_dependent_C is not None:
            C_mean_samples = jnp.tile(C[0], (n_samples_obs_pref_s,1) )
        if sys_dependent_C is not None:
            # Sample preference priors
            key, use_key = random.split(key)
            sample_sys = random.multivariate_normal(use_key, belief_sys_params_prior[0], belief_sys_params_prior[1], shape=(n_samples_obs_pref_s,))
            for i in range(n_samples_obs_pref_s):
                C_mean_samples = C_mean_samples.at[i, sys_dependent_C[0]].set(sample_sys[i, sys_dependent_C[1]])

        if state_dependent_C is not None:
            key, use_key = random.split(key)
            sample_state_for_C = random.multivariate_normal(use_key, belief_state_prior[0], belief_state_prior[1], shape=(n_samples_obs_pref_s,))
            
            for i in range(n_samples_obs_pref_s):
                C_mean_samples = C_mean_samples.at[i, state_dependent_C[0]].set(sample_state_for_C[i, state_dependent_C[1]])    

        # parallelized function
        def calc_nefe(pi, key):
            def rollout_step(i, carry, pi):
                key = carry[0]
                nefes = carry[1]
                pragmatics = carry[2]
                info_gains = carry[3]
                belief_state = carry[4]
                belief_noise = belief_noise_prior
                belief_sys = belief_sys_params_prior

                a = pi[i] # selected action in step i
                nefe = 0 # negative expected free energy for this timestep
                pragmatic = 0 # pragmatic value for this timestep
                info_gain = 0 # information gain for this timestep

                # Where will I be after taking action a for multistep steps?
                belief_state_pred = belief_state
                for _ in range(multistep):
                    key, use_key = random.split(key)
                    belief_state_pred, _ = _update_belief_a(belief_state_pred, belief_noise, belief_sys, a=a, key=use_key) 

                if use_info_gain or use_observation_preference:
                    if use_observation_preference:
                        # New version for dim_obs >= 1
                        key, use_key = random.split(key)
                        sample_states = random.multivariate_normal(use_key, *belief_state_pred, shape=(n_samples_obs_pref_s,))  # sample from state belief

                        key, use_key = random.split(key)
                        _, obs_variance = sample_obs_noise_params(belief_noise, n_samples_obs_pref_s, use_key)

                        # Sample system parameters
                        key, use_key = random.split(key)
                        sample_sys = random.multivariate_normal(use_key, belief_sys[0], belief_sys[1], shape=(n_samples_obs_pref_s,))

                        # Halluzinate observations
                        oo_noise_free = jnp.apply_along_axis(lambda xsys: get_observation_complete(xsys[:dim_state], *xsys[dim_state:]), 1, jnp.hstack([sample_states, sample_sys])) # sampled observations
                        
                        def halluzinate_obs(o_noise_free, obs_variance_single, key):
                            if has_observation_noise:
                                hal_o = random.multivariate_normal(key, o_noise_free, jnp.diag(obs_variance_single), shape=(n_samples_obs_pref_o,))
                            else:
                                hal_o = jnp.tile(o_noise_free, (n_samples_obs_pref_o,1))
                            return hal_o
                        keys = random.split(key, num=n_samples_obs_pref_s+1)
                        key = keys[0]
                        batch_keys = keys[1:]
                        oo_pref = vmap(lambda o_noise_free, obs_variance_single, key: halluzinate_obs(o_noise_free, obs_variance_single, key), in_axes=(0,0,0), out_axes=0)(oo_noise_free, obs_variance, batch_keys).reshape(-1,dim_observation)

                        # Do I like observing this?
                        oo_pref = oo_pref[:,C_index]
                        if sys_dependent_C is not None or state_dependent_C is not None:
                            pragmatic =  jnp.mean(jnp.apply_along_axis(lambda C_mean: logpdf(oo_pref, C_mean, C[1]).mean(), 1, C_mean_samples))
                        else:
                            pragmatic = logpdf(oo_pref, *C).mean()
                        
                        nefe += scale_pragmatic_value * pragmatic

                    if use_info_gain:
                        if n_samples_ig_s < 1 or n_samples_ig_o < 1:
                            # Sample system parameters
                            key, use_key = random.split(key)
                            sample_sys = random.multivariate_normal(use_key, belief_sys[0], belief_sys[1])
                            # Use mean to calculate info gain
                            o_mean = get_observation_complete(belief_state_pred[0], *sample_sys) # sampled observation
                            key, use_key = random.split(key)
                            belief_state_o, _, _ = _update_belief(belief_state_pred, belief_noise, belief_sys, o_mean, key=use_key)
                            info_gain = kl_jax(*belief_state_o, *belief_state)
                        else:
                            if n_samples_ig_s * n_samples_ig_o > n_samples_obs_pref_s * n_samples_obs_pref_o:
                                # New version for dim_obs >= 1
                                key, use_key = random.split(key)
                                sample_states = random.multivariate_normal(use_key, *belief_state_pred, shape=(n_samples_ig_s,))  # sample from state belief

                                key, use_key = random.split(key)
                                _, obs_variance = sample_obs_noise_params(belief_noise, n_samples_ig_s, use_key)
                                # Sample system parameters
                                key, use_key = random.split(key)
                                sample_sys = random.multivariate_normal(use_key, belief_sys[0], belief_sys[1], shape=(n_samples_ig_s,))
                                # Halluzinate observations
                                oo_noise_free = jnp.apply_along_axis(lambda xsys: get_observation_complete(xsys[:dim_state], *xsys[dim_state:]), 1, jnp.hstack([sample_states, sample_sys])) # sampled observations
                                def halluzinate_obs(o_noise_free, obs_variance_single, key):
                                    hal_o = random.multivariate_normal(key, o_noise_free, jnp.diag(obs_variance_single), shape=(n_samples_obs_pref_o,))
                                    return hal_o
                                keys = random.split(key, num=n_samples_ig_s+1)
                                key = keys[0]
                                batch_keys = keys[1:]
                                oo = vmap(lambda o_noise_free, obs_variance_single, key: halluzinate_obs(o_noise_free, obs_variance_single, key), in_axes=(0,0,0), out_axes=0)(oo_noise_free, obs_variance, batch_keys).reshape(-1,dim_observation)
                            else:
                                # Use the same observations as for observation preference
                                # Select a subset of the observations
                                key, use_key = random.split(key)
                                oo = random.choice(use_key, oo_pref, shape=(n_samples_ig_s * n_samples_ig_o,), replace=False) # select n_samples_ig_s * n_samples_ig_o observations from the observations halluzinated for the observation preference

                            # # Do I learn about s from observing o?
                            def calc_info_gain(o, key):
                                belief_state_o, _, _ = _update_belief(belief_state_pred, belief_noise, belief_sys, o, key=key)
                                kl_state = kl_jax(*belief_state_o, *belief_state)
                                return kl_state 
                            
                            keys = random.split(key, num=n_samples_ig+1)
                            key = keys[0]
                            batch_keys = keys[1:]
                            kl_o = vmap(calc_info_gain, in_axes=(0,0), out_axes=0)(oo, batch_keys) #TODO: This step is slow, can we speed it up?
                            info_gain = jnp.mean(kl_o)
                        nefe += info_gain

                elif not use_observation_preference:
                    # Use state-based preference distribution
                    # Do I like being there?
                    pragmatic = _calc_pragmatic_value(belief_state_pred, C=C, params=params)
                    nefe += scale_pragmatic_value * pragmatic

                # concatenate expected free energy across future time steps
                return (key, nefes.at[i].set(nefe), pragmatics.at[i].set(pragmatic), info_gains.at[i].set(info_gain), belief_state_pred) #TODO check if theta_pred or theta_o
                ## End rollout_step
            
            key, use_key = random.split(key)
            _, step_nefes, step_pragmatics, step_info_gains, _ = fori_loop(0, horizon, lambda i, carry: rollout_step(i, carry, pi), (use_key, jnp.zeros((horizon,)), jnp.zeros((horizon,)), jnp.zeros((horizon,)), belief_state_prior))
           
            return step_nefes.mean(), step_pragmatics.mean(), step_info_gains.mean() # expected value over steps
            ## End calc_nefe

        # evaluate negative expected free energy of all plans in parallel
        keys = random.split(key, num=n_plans+1)
        key = keys[0]
        batch_keys = keys[1:]
        nefes, pragmatics, info_gains = vmap(lambda pi, key: calc_nefe(pi, key), in_axes=(0,0), out_axes=(0,0,0))(plans, batch_keys)

        # compute probability of following each plan
        p_pi = softmax_jax(nefes)

        if select_max_pi:
            plani = jnp.argmax(nefes)
        else:
            key, use_key = random.split(key)
            plani = random.choice(use_key, n_plans, p=p_pi)

        sel_plan = plans[plani]
        nefe_plan = nefes[plani]
        pragmatic_plan = pragmatics[plani]
        info_gain_plan = info_gains[plani]
        return sel_plan, nefe_plan, pragmatic_plan, info_gain_plan, plans, nefes, pragmatics, info_gains

class AIF_Simulation:
    ''' Class to run Active Inference simulations with a given agent and generative process
    '''
    def __init__(self, agent, generative_process, noise_params=None):
        self.agent = agent
        self.generative_process = generative_process
        self.noise_params = {}
        if noise_params is not None:
            noise_params, _ = refactor_noise_params(noise_params)
            self.noise_params.update(noise_params)
        self.dim_actions = self.generative_process.dim_action
        self.noise_params['dim_action'] = self.dim_actions
        self.agent.initialize()

    def sample_o_given_s(self,s,noise_params,key): 
        ''' Function to sample the observation o given the state s
        Args:
            s: current state
            noise_params: noise parameters
            key: unused jax random key
        Returns:
            o: sampled observation
        '''
        # Generate observation from any given state s
        obs = self.generative_process._get_observation(s)
        dim_observation = self.generative_process.dim_observation
        key, use_key = random.split(key)
        if 'observation_std' in noise_params:
            num_noise_vals = noise_params['observation_std'][1].shape[0]
            obs_indices = noise_params['observation_std'][1]
            noise_std = noise_params['observation_std'][2]
            if num_noise_vals > 1:
                obs += jnp.zeros(dim_observation).at[obs_indices].set(random.multivariate_normal(use_key, jnp.zeros((num_noise_vals,)), jnp.diag(jnp.square(noise_std))))
            else:
                obs += jnp.zeros(dim_observation).at[obs_indices].set(random.normal(use_key) * noise_std)
        key, use_key = random.split(key)
        if 'state_dependent_obs_noise' in noise_params:
            num_noise_vals = noise_params['state_dependent_obs_noise'][1].shape[0]
            obs_indices = noise_params['state_dependent_obs_noise'][1]
            state_indices = noise_params['state_dependent_obs_noise'][2]
            noise_std = noise_params['state_dependent_obs_noise'][3]
            if num_noise_vals > 1:
                obs += jnp.zeros(dim_observation).at[obs_indices].set(random.multivariate_normal(use_key, jnp.zeros((num_noise_vals,)), jnp.diag(jnp.square(jnp.multiply(EPS+s[state_indices]), noise_std))))
            else:
                obs += jnp.zeros(dim_observation).at[obs_indices].set(random.normal(use_key) * (EPS+s[state_indices]) * noise_std)

        return obs 
    
    def sample_o(self, key): 
        ''' Function to sample the observation o given the current state of the generative process
        Args:
            self: AIF_Simulation object
            key:  unused jax random key
        Returns:
            o: sampled observation
        '''
        return self.sample_o_given_s(self.generative_process.x, self.noise_params, key=key) # Sample observation from current state of the generative process
    
    def apply_control_noise(self, a, key):
        ''' Function to apply control noise to the action a
        Args:
            a: action
            key: unused jax random key
        Returns:
            a: action with added noise
        '''
        if 'signal_dependent_noise' in self.noise_params:
            noise_indices = self.noise_params['signal_dependent_noise'][0]
            key, use_key = random.split(key)
            d1 = random.multivariate_normal(use_key, jnp.zeros((noise_indices.shape[0],)), jnp.diag(jnp.square(self.noise_params['signal_dependent_noise'][2])))
        else:
            d1 = jnp.zeros((0,))
        if 'constant_motor_noise' in self.noise_params:
            noise_indices = self.noise_params['constant_motor_noise'][0]
            key, use_key = random.split(key)
            d2 = random.multivariate_normal(use_key, jnp.zeros((noise_indices.shape[0],)), jnp.diag(jnp.square(self.noise_params['constant_motor_noise'][2])))
        else:
            d2 = jnp.zeros((0,))
        d = jnp.concatenate((d1, d2), axis=0)  # Concatenate noise vectors
        a = self.agent._apply_control_noise_given_d(a, d, self.noise_params)  # Apply control noise to the action
        return a

    
    def step(self, a, debug=False, key=None):
        ''' Function to step the generative process given an action a
        Args:
            self: AIF_Simulation object
            a: action
            debug: If true, additional debug information is returned
            key: unused jax random key
        Returns:
            o: sampled observation
        '''
        key, use_key = random.split(key)
        a_applied = self.apply_control_noise(a, use_key)
        # key, use_key = random.split(key)
        self.generative_process.step(u=a_applied) # forward generative process
        if debug:
            return self.sample_o(key), self.generative_process.x, a_applied
        return self.sample_o(key) # return observation 
    
    def change_noise_params(self, noise_params):
        ''' Function to change the parameters of the agent
        Args:
            self: AIF_Simulation object
            noise_params: new noise parameters
        '''
        self.noise_params, _ = refactor_noise_params(noise_params)
        self.noise_params['dim_action'] = self.dim_actions

    def reset(self):
        ''' Function to reset the generative process and agent
        Args:
            self: AIF_Simulation object
        '''
        self.generative_process.reset()
        self.agent.reset()

    ## Different run function
    def run_inference_only(self, numsteps=100, a=None, random_a=False, reset=True, key=random.key(42)):
        agent = self.agent

        if a == None:
            a = agent.params['a_lims'][1] # Default maximum action

        belief_state = agent.belief_state

        if reset:
            self.reset()

        # Logging
        bb = [belief_state]
        xx = [self.generative_process.x] # history of system states
        oo = []
        aa = []
        aa_applied = []
        lll = []
        LR = []
        # Start interaction loop
        for i in tqdm(range(numsteps)):
            # Select random action
            if random_a:
                key, use_key = random.split(key)
                a = random.uniform(use_key, shape=(agent.params['dim_action'],), minval=agent.params['a_lims'][0], maxval=agent.params['a_lims'][1])
            
            # Make system step
            key, use_key = random.split(key)
            o, x, a_applied = self.step(a, debug=True, key=use_key)
        
            # Update belief state with action
            key, use_key = random.split(key)
            belief_state, _ = agent.update_belief_state(belief_state, agent.belief_noise, agent.belief_sys, a, key=use_key)

            # # Update belief state with observation
            key, use_key = random.split(key)
            belief_state, ll, lr = agent.update_belief_state_obs(belief_state, agent.belief_noise, agent.belief_sys, o, key=use_key)
            
            # Logging
            xx.append(x)
            aa.append(a)
            aa_applied.append(a_applied)
            oo.append(o)
            bb.append(belief_state)
            lll.append(ll)
            LR.append(lr)

            if jnp.isnan(belief_state[0]).any() or np.isnan(belief_state[1]).any():
                print("NaN in belief state or covariance matrix detected. Stopping simulation.")
                break
        return bb, xx, oo, aa, aa_applied, lll, LR
    
    def run_aif_perceptual_delay(self, numsteps=100, break_criteria=None, reset=True, sys_belief_after_rt=None, key=random.key(42)):
        agent = self.agent

        reaction_time_steps = int(agent.params['reaction_time']//agent.dt)

        belief_state = agent.belief_state

        print(f"DEBUG: Running simulation with initial system belief: {agent.belief_sys}")
        print(f"DEBUG: Reaction time: {agent.params['reaction_time']}")
        action_buffer = jnp.zeros((reaction_time_steps, agent.params['dim_action']))
        observation_buffer = jnp.zeros((reaction_time_steps, agent.params['dim_observation']))

        if reset:
            self.reset()

        # Logging
        bb = [belief_state]
        bb_after_rt = []

        lll = []
        LR = []
        xx = [self.generative_process.x] # history of system states
        oo = []
        aa = []
        aa_applied = []
        NEFE_PLAN = []
        PRAGMATIC_PLAN = []
        INFO_GAIN_PLAN = []
        NEFES = []
        PRAGMATICS = []
        INFO_GAINS = []
        for i in tqdm(range(numsteps)):
            # Let the agent know about the target after reaction time
            if i == reaction_time_steps and sys_belief_after_rt is not None:
                agent.belief_sys = sys_belief_after_rt

            # Predict the state after reaction time
            belief_state_after_rt = belief_state
            if reaction_time_steps > 0:
                for j in range(reaction_time_steps):
                    a = action_buffer[j]
                    key, use_key = random.split(key)
                    belief_state_after_rt, BS = agent.update_belief_state(belief_state_after_rt, agent.belief_noise, agent.belief_sys, a, key=use_key)
            bb_after_rt.append(belief_state_after_rt)

            # Select action as usual
            key, use_key = random.split(key)
            sel_plan, nefe_plan, pragmatic_plan, info_gain_plan, plans, nefes, pragmatics, info_gains = agent.select_action(belief_state_after_rt, agent.belief_noise, agent.belief_sys, key=use_key)
            a_new = sel_plan[0]

            # Make system step
            key, use_key = random.split(key)
            o, x, a_applied = self.step(a_new, debug=True, key=use_key)

            # Fill observation buffer with the observation
            if reaction_time_steps > 0:
                observation_buffer = jnp.roll(observation_buffer, -1, axis=0)
                observation_buffer = observation_buffer.at[-1].set(o)

            ## LOGGING
            NEFE_PLAN.append(nefe_plan)
            PRAGMATIC_PLAN.append(pragmatic_plan)
            INFO_GAIN_PLAN.append(info_gain_plan)
            NEFES.append(nefes)
            PRAGMATICS.append(pragmatics)
            INFO_GAINS.append(info_gains)
            xx.append(x)
            aa.append(a_new)
            aa_applied.append(a_applied)
            oo.append(o)

            # Update using buffered action
            if reaction_time_steps > 0:
                a = action_buffer[0]
            else:
                a = a_new
            key, use_key = random.split(key)
            belief_state, BS = agent.update_belief_state(belief_state, agent.belief_noise, agent.belief_sys, a, key=use_key)

            # Fill action buffer with the selected action
            if reaction_time_steps > 0:
                action_buffer = jnp.roll(action_buffer, -1, axis=0)
                action_buffer = action_buffer.at[-1].set(a_new)

            # After initial reaction time, update belief using the buffered observation
            if i >= (reaction_time_steps-1):
                if reaction_time_steps > 0:
                    o = observation_buffer[0]
                key, use_key = random.split(key)
                belief_state, ll, lr = agent.update_belief_state_obs(belief_state, agent.belief_noise, agent.belief_sys,  o, key=use_key)
                lll.append(ll)
                LR.append(lr)

            bb.append(belief_state)

            if jnp.isnan(belief_state[0]).any() or np.isnan(belief_state[1]).any():
                print("NAN in belief. Breaking...")
                break

            if break_criteria is not None:
                if break_criteria(belief_state, x, o, a_applied, observation_buffer, action_buffer, i):
                    print("Break criteria met. Stopping simulation.")
                    break
        return bb, bb_after_rt, xx, oo, aa, aa_applied, lll, LR, NEFE_PLAN, PRAGMATIC_PLAN, INFO_GAIN_PLAN, NEFES, PRAGMATICS, INFO_GAINS
