from typing import Dict, List, Optional, Tuple, Callable
import warnings

import numpy as np
import torch

import mbrl.models.util as model_util
import mbrl.types
import mbrl.util.math

from mbrl.models import Model, OneDTransitionRewardModel



#TODO: could be modularized for any "hybrid" model
class OneDTransitionRewardModelBikes(OneDTransitionRewardModel):
    """Wrapper class for 1-D dynamics models specific for the Bikes environment

        We only care about learning the dynamics of the bikes distribution after
        taking an action.
        To be more precise, at each environment step, we start with a given distribution of the bikes,
        according to the date and time of the day, we choose to add some bikes thanks to the trucks in 
        a given set of centroids. So far nothing has to be learned, the dynamics is known. 
        But then how the bikes will be used over the map is in general unknown.
    """

    def __init__(
        self,
        model: Model,
        transform_obs: Callable,
        transform_act: Callable,
        target_is_delta: bool = True,
        normalize: bool = False,
        normalize_double_precision: bool = False,
        learned_rewards: bool = True,
        obs_process_fn: Optional[mbrl.types.ObsProcessFnType] = None,
        no_delta_list: Optional[List[int]] = None,
        num_elites: Optional[int] = None,
    ):
        super().__init__(model, target_is_delta, normalize, normalize_double_precision, learned_rewards, obs_process_fn, no_delta_list, num_elites)
        self.transform_obs = transform_obs
        self.transform_act = transform_act
        self.obs_process_fn = self._get_new_obs

    def _get_new_obs(self, batch_obs, batch_action, exclude_keys = ["bikes_dist_after_shift"]):
        """
        Compute the new observation after taking the action a,
        but before the rentals simulation

        Function used to preprocess the obs when calling _get_model_input
        """

        if len(batch_obs.shape) == 1:
            batch_obs = np.expand_dims(batch_obs)
        all_obs = []
        for obs, action in zip(batch_obs, batch_action):

            action = self.transform_act(action)
            obs = self.transform_obs(obs)

            bikes_distr_shape = obs["bikes_dist_before_shift"].shape
            delta_bikes = np.zeros(bikes_distr_shape, dtype=int)
            truck_centroid = action["truck_centroid"]
            truck_num_bikes = action["truck_num_bikes"]
            num_trucks = len(truck_centroid)
            for truck in range(num_trucks):
                delta_bikes[int(truck_centroid[int(truck)])] += truck_num_bikes[int(truck)]
            obs["bikes_dist_before_shift"] = obs["bikes_dist_after_shift"] + delta_bikes
            obs["bikes_dist_after_shift"] = obs["bikes_dist_before_shift"]

            obs = np.concatenate([value for key, value in obs.items() if key not in exclude_keys])
            all_obs.append(obs)

        all_obs = np.array(all_obs)
        if len(all_obs.shape) == 1:
            all_obs = np.expand_dims(all_obs)
        
        return all_obs
    
    def _get_next_obs(self, batch_next_obs):

        if len(batch_next_obs.shape) == 1:
            batch_next_obs = np.expand_dims(batch_next_obs)
        all_next_obs = []
        for next_obs in batch_next_obs:
            next_obs = self.transform_obs(next_obs)["bikes_dist_after_shift"]
            all_next_obs.append(next_obs)

        return np.array(all_next_obs)
    
    def _get_model_input(
        self,
        obs: mbrl.types.TensorType,
        action: mbrl.types.TensorType,
    ) -> torch.Tensor:
        if self.obs_process_fn:
            obs = self.obs_process_fn(obs, action)
        obs = model_util.to_tensor(obs).to(self.device)
        # action = model_util.to_tensor(action).to(self.device)
        # model_in = torch.cat([obs, action], dim=obs.ndim - 1)
        model_in = obs
        if self.input_normalizer and False: #TODO: remettre normalizer asap !!!!
            # Normalizer lives on device
            model_in = self.input_normalizer.normalize(model_in)
        model_in = model_in.float().to(self.device)
        return model_in

    def _process_batch(
        self, batch: mbrl.types.TransitionBatch, _as_float: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        obs, action, next_obs, reward, _, _ = batch.astuple()
        if self.target_is_delta:
            target_obs = next_obs - obs
            for dim in self.no_delta_list:
                target_obs[..., dim] = next_obs[..., dim]
        else:
            target_obs = next_obs
        target_obs = self._get_next_obs(target_obs)
        target_obs = model_util.to_tensor(target_obs).to(self.device)

        model_in = self._get_model_input(obs, action)
        if self.learned_rewards:
            reward = model_util.to_tensor(reward).to(self.device).unsqueeze(reward.ndim)
            target = torch.cat([target_obs, reward], dim=obs.ndim - 1)
        else:
            target = target_obs
        return model_in.float(), target.float()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, ...]:
        """Calls forward method of base model with the given input and args."""


        model_input = self._get_new_obs(x.numpy())

        new_bike_distribution = self.model.forward(model_input, *args, **kwargs).numpy()

        out = self.transform_obs(model_input)

        out["bikes_dist_after_shift"] = new_bike_distribution
        out["time_counter"] += 1

        return torch.from_numpy(self.transform_obs(out))

    def update_normalizer(self, batch: mbrl.types.TransitionBatch):
        """Updates the normalizer statistics using the batch of transition data.

        The normalizer will compute mean and standard deviation the obs and action in
        the transition. If an observation processing function has been provided, it will
        be called on ``obs`` before updating the normalizer.

        Args:
            batch (:class:`mbrl.types.TransitionBatch`): The batch of transition data.
                Only obs and action will be used, since these are the inputs to the model.
        """
        #TODO: Should we change something here ??
        #YESSS, but have to change self.input_normalizer
        return
        if self.input_normalizer is None:
            return
        warnings.warn("This function might not be accurate !!")
        obs, action = batch.obs, batch.act
        if obs.ndim == 1:
            obs = obs[None, :]
            action = action[None, :]
        if self.obs_process_fn:
            obs = self.obs_process_fn(obs, action)
        model_in_np = np.concatenate([obs, action], axis=obs.ndim - 1)
        self.input_normalizer.update_stats(model_in_np)

    def get_output_and_targets(
        self, batch: mbrl.types.TransitionBatch
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        """Returns the model output and the target tensors given a batch of transitions.

        This method constructs input and targets from the information in the batch,
        then calls `self.model.forward()` on them and returns the value.
        No gradient information will be kept.

        Args:
            batch (transition batch): a batch of transition to train the model.

        Returns:
            (tuple(tensor), tensor): the model outputs and the target for this batch.
        """
        with torch.no_grad():
            model_in, target = self._process_batch(batch)
            new_bike_distributions = self.model.forward(model_in).numpy()
            all_out = []
            for input_, new_bike_distr in zip(model_in, new_bike_distributions):
                out = self.transform_obs(input_)
                out["bikes_dist_after_shift"] = new_bike_distr
                out["time_counter"] += 1
                all_out.append(self.transform_obs(out))

        return torch.from_numpy(all_out), target

    def sample(
        self,
        act: torch.Tensor,
        model_state: Dict[str, torch.Tensor],
        deterministic: bool = False,
        rng: Optional[torch.Generator] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[Dict[str, torch.Tensor]],
    ]:
        """Samples next observations and rewards from the underlying 1-D model.

        This wrapper assumes that the underlying model's sample method returns a tuple
        with just one tensor, which concatenates next_observation and reward.

        Args:
            act (tensor): the action at.
            model_state (tensor): the model state st.
            deterministic (bool): if ``True``, the model returns a deterministic
                "sample" (e.g., the mean prediction). Defaults to ``False``.
            rng (random number generator): a rng to use for sampling.

        Returns:
            (tuple of two tensors): predicted next_observation (o_{t+1}) and rewards (r_{t+1}).
        """
        obs = model_util.to_tensor(model_state["obs"]).to(self.device)
        new_obs = self.obs_process_fn(obs, act, exclude_keys=[])
        model_in = self._get_model_input(model_state["obs"], act)
        if not hasattr(self.model, "sample_1d"):
            raise RuntimeError(
                "OneDTransitionRewardModel requires wrapped model to define method sample_1d"
            )
        preds, next_model_state = self.model.sample_1d(
            model_in, model_state, rng=rng, deterministic=deterministic
        )
        next_observs = preds[:, :-1] if self.learned_rewards else preds

        new_bike_distributions = next_observs.numpy()
        all_out = []
        for obs, new_bike_distr in zip(new_obs, new_bike_distributions):
            out = self.transform_obs(obs)
            out["bikes_dist_after_shift"] = new_bike_distr
            out["time_counter"] += 1
            all_out.append(self.transform_obs(out))
        next_observs = torch.from_numpy(np.array(all_out))

        if self.target_is_delta:
            tmp_ = next_observs + obs
            for dim in self.no_delta_list:
                tmp_[:, dim] = next_observs[:, dim]
            next_observs = tmp_
        rewards = preds[:, -1:] if self.learned_rewards else None
        next_model_state["obs"] = next_observs
        end = time()
        print(2)
        print(end-start)
        return next_observs, rewards, None, next_model_state
    
    def reset(
        self, obs: torch.Tensor, rng: Optional[torch.Generator] = None
    ) -> Dict[str, torch.Tensor]:

        if not hasattr(self.model, "reset_1d"):
            raise RuntimeError(
                "OneDTransitionRewardModel requires wrapped model to define method reset_1d"
            )
        obs = model_util.to_tensor(obs).to(self.device)
        model_state = {"obs": obs}
        model_state.update(self.model.reset_1d(obs, rng=rng))

        return model_state
