from typing import Any, Dict, List, Optional, Tuple, Callable
import warnings

import numpy as np
import torch
from torcheval.metrics.functional import r2_score


import mbrl.models.util as model_util
import mbrl.types
import mbrl.util.math

from mbrl.models import Model, OneDTransitionRewardModel


class OneDTransitionRewardModelDictSpace(OneDTransitionRewardModel):
    """
    Wrapper class for 1-D dynamics models when dealing with a DictSpacesEnv.
    In this case part of the whole dynamics are known and thus only part of the
    observation and action are required to learn the unknown dynamics.
    e.g. In the Bikes environment, we know the dynamics of adding bikes at different locations
    but we want to learn how the trips will occur during a given period after adding the bikes.
    Then the dynamics of stepping day, month and timeshift is also known.
    """

    # TODO: Deal with normalize and rescaling (for now none it used as either bad implemented
    # or harder to train on..)

    def __init__(
        self,
        model: Model,
        map_obs: Dict,
        map_act: Dict,
        rescale_obs: Callable,
        rescale_act: Callable,
        model_input_obs_key: List = None,
        model_input_act_key: List = None,
        model_output_key: List = None,
        target_is_delta: bool = True,
        normalize: bool = False,
        rescale_input: bool = False,
        normalize_double_precision: bool = False,
        learned_rewards: bool = True,
        obs_preprocess_fn: Optional[mbrl.types.ObsProcessFnType] = None,
        obs_postprocess_fn: Optional[mbrl.types.ObsProcessFnType] = None,
        no_delta_list: Optional[List[int]] = None,
        num_elites: Optional[int] = None,
    ):
        super().__init__(
            model,
            target_is_delta,
            normalize,
            normalize_double_precision,
            learned_rewards,
            obs_preprocess_fn,
            no_delta_list,
            num_elites,
        )
        self.map_obs = map_obs
        self.map_act = map_act
        self.obs_length = self.map_obs["length"]
        self.act_length = self.map_act["length"]
        self.obs_process_fn = obs_preprocess_fn
        self.obs_postprocess_fn = obs_postprocess_fn

        if model_input_obs_key is None:
            model_input_obs_key = [
                key for key in self.map_obs.keys() if key != "length"
            ]
        if model_input_act_key is None:
            model_input_act_key = [
                key for key in self.map_act.keys() if key != "length"
            ]
        if model_output_key is None:
            model_output_key = [key for key in self.map_obs.keys() if key != "length"]

        # Create the corresponding masks and input_size
        self.in_size = 0
        self.model_input_mask = np.zeros(self.obs_length)
        for key in model_input_obs_key:
            self.model_input_mask[self.map_obs[key]] = 1
        self.in_size += np.count_nonzero(self.model_input_mask)
        self.model_input_mask = np.ma.make_mask(self.model_input_mask)

        model_input_act_mask = np.zeros(self.act_length)
        for key in model_input_act_key:
            model_input_act_mask[self.map_act[key]] = 1
        self.in_size += np.count_nonzero(model_input_act_mask)
        model_input_act_mask = np.ma.make_mask(model_input_act_mask, shrink=False)
        self.model_input_mask = np.concatenate(
            (self.model_input_mask, model_input_act_mask)
        )

        self.model_output_mask = np.zeros(self.obs_length)
        model_output_length = 0
        for key in model_output_key:
            model_output_length += self.map_obs[key].stop - self.map_obs[key].start
            self.model_output_mask[self.map_obs[key]] = 1
        self.model_output_mask = np.ma.make_mask(self.model_output_mask)

        # Reinitialize the output with the output_size
        if self.learned_rewards:
            model_output_length += 1

        # TODO: remove this line
        self.output_normalizer = None
        self.rescale_input = rescale_input
        if self.rescale_input:
            self.rescale_obs = rescale_obs
            self.rescale_act = rescale_act
        if normalize:
            self.output_normalizer = mbrl.util.math.Normalizer(
                model_output_length,
                self.model.device,
                dtype=torch.double if normalize_double_precision else torch.float,
            )

    def _get_next_obs(self, batch_next_obs: mbrl.types.TensorType):
        if len(batch_next_obs.shape) == 1:
            batch_next_obs = np.expand_dims(batch_next_obs, axis=0)
        if not np.any(self.model_output_mask):
            return None
        else:
            return batch_next_obs[..., self.model_output_mask]

    def _get_model_input(
        self, obs: mbrl.types.TensorType, action: mbrl.types.TensorType
    ) -> torch.Tensor:
        if self.obs_process_fn:
            obs = self.obs_process_fn(obs, action)
        obs = model_util.to_tensor(obs).to(self.device)
        action = model_util.to_tensor(action).to(self.device)
        if self.rescale_input:
            obs = self.rescale_obs(obs)
            action = self.rescale_act(action)
        model_in = torch.cat([obs, action], dim=obs.ndim - 1)

        model_in = model_in.float().to(self.device)
        masked_model_in = model_in[..., self.model_input_mask]
        return masked_model_in, obs

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
        target_obs = self._get_next_obs(target_obs)  # Only change in this function
        if target_obs is not None:
            target_obs = model_util.to_tensor(target_obs).to(self.device)

        model_in, _ = self._get_model_input(obs, action)
        if self.learned_rewards:
            reward = model_util.to_tensor(reward).to(self.device).unsqueeze(reward.ndim)
            if target_obs is not None:
                target = torch.cat([target_obs, reward], dim=obs.ndim - 1)
            else:
                target = reward
        else:
            target = target_obs

        if target is None:
            raise ValueError("You try to predict nothing")

        if self.output_normalizer:
            target = self.output_normalizer.normalize(target)

        return model_in.float(), target.float()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, ...]:
        """Calls forward method of base model with the given input and args."""

        warnings.warn("Not used so far, make sure it works")

        if len(x.shape) == 1:
            x = x[None, ...]

        obs = x[: self.obs_length]
        action = x[self.obs_length : self.obs_length + self.act_length]
        model_input = self._get_model_input(obs, action)
        output = obs
        sub_output = self.model.forward(model_input, *args, **kwargs).numpy()
        output[self.model_output_mask] = sub_output
        output = self.obs_postprocess_fn(output)
        # TODO: Denormalize output ???
        return output

    def update_normalizer(self, batch: mbrl.types.TransitionBatch):
        """Updates the normalizer statistics using the batch of transition data.

        The normalizer will compute mean and standard deviation of the next_obs and reward in
        the transition.

        Args:
            batch (:class:`mbrl.types.TransitionBatch`): The batch of transition data.
                Only next_obs and reward will be used, since these are the outputs to the model.
        """
        if self.output_normalizer is None:
            return
        obs, next_obs, reward = batch.obs, batch.next_obs, batch.rewards
        if self.target_is_delta:
            target_obs = next_obs - obs
            for dim in self.no_delta_list:
                target_obs[..., dim] = next_obs[..., dim]
        else:
            target_obs = next_obs
        target_obs = self._get_next_obs(target_obs)
        if reward.ndim == 1:
            reward = np.expand_dims(reward, axis=-1)
        if target_obs is None:
            target = reward
        else:
            if target_obs.ndim == 1:
                target_obs = np.expand_dims(target_obs, axis=-1)
            if self.learned_rewards:
                target = np.concatenate([target_obs, reward], axis=obs.ndim - 1)
            else:
                target = target_obs
        self.output_normalizer.update_stats(target)

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
        warnings.warn("Not used so far, make sure it works")
        with torch.no_grad():
            model_in, target = self._process_batch(batch)
            sub_output = self.model.forward(model_in)
            if self.output_normalizer:
                sub_output = self.output_normalizer.denormalize(sub_output)

            out = model_in
            out[..., self.model_output_mask] = sub_output
            out = self.obs_postprocess_fn(out)

            return out, target

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
        model_in, preprocessed_obs = self._get_model_input(model_state["obs"], act)
        if not hasattr(self.model, "sample_1d"):
            raise RuntimeError(
                "OneDTransitionRewardModel requires wrapped model to define method sample_1d"
            )
        preds, next_model_state = self.model.sample_1d(
            model_in, model_state, rng=rng, deterministic=deterministic
        )
        if self.output_normalizer is not None:
            preds = self.output_normalizer.denormalize(preds).float()
        next_observs = preds[:, :-1] if self.learned_rewards else preds

        next_obs = preprocessed_obs
        if next_observs.shape[-1]>0:
            next_obs[:, self.model_output_mask] = next_observs
        next_obs = self.obs_postprocess_fn(next_obs)

        if self.target_is_delta:
            tmp_ = next_obs + obs
            for dim in self.no_delta_list:
                tmp_[:, dim] = next_obs[:, dim]
            next_obs = tmp_
        rewards = preds[:, -1:] if self.learned_rewards else None
        next_model_state["obs"] = next_obs

        return next_obs, rewards, None, next_model_state

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
