import torch as tc

from drl.algos.abstract import Algo


class PPO(Algo):
    def __init__(self, rank, config):
        super().__init__(rank, config)
        self._learning_system = self._get_learning_system()

    def _get_learning_system(self):
        env_config = self._config.get('env')
        env = self._get_env(env_config)

        policy_config = self._config.get('policy_net')
        policy_net = self._get_net(policy_config)
        policy_optimizer_config = policy_config.get('optimizer')
        policy_optimizer = self._get_opt(policy_optimizer_config, policy_net)

        value_config = self._config.get('value_net')
        value_net, value_optimizer = None, None
        if not value_config.get('use_shared_architecture'):
            value_net = self._get_net(value_config)
            value_optimizer_config = value_config.get('optimizer')
            value_optimizer = self._get_opt(value_optimizer_config, value_net)

        checkpointables = {
            'policy_net': policy_net,
            'policy_optimizer': policy_optimizer,
            'value_net': value_net,
            'value_optimizer': value_optimizer
        }
        checkpointables_ = {k: v for k,v in checkpointables.items()}
        checkpointables_.update(env.get_checkpointables())
        global_step = self._maybe_load_checkpoints(checkpointables_, step=None)
        return {'global_step': global_step, 'env': env, **checkpointables}

    @tc.no_grad()
    def _annotate(self, policy_net, value_net, trajectory):
        algo_config = self._config.get('algo')

        # get trajectory variables.
        observations = trajectory.get('observations')
        actions = trajectory.get('actions')
        rewards = trajectory.get('rewards')
        dones = trajectory.get('dones')

        # decide who should predict what.
        reward_keys = rewards.keys()
        relevant_reward_keys = [k for k in reward_keys if k != 'extrinsic_raw']
        policy_predict = ['policy']
        value_predict = []
        for key in relevant_reward_keys:
            value_predict.append(f'value_{key}')
        if value_net is None:
            policy_predict.extend(value_predict)

        # compute logprobs of actions.
        predictions = policy_net(observations, policy_predict)
        logprobs = predictions.get('policy').log_prob(actions)
        logprobs = logprobs[0:-1]

        # compute value estimates.
        if value_net is None:
            vpreds = {k: predictions[k] for k in value_predict}
        else:
            vpreds = value_net(observations, value_predict)
        vpreds = {k.partition('_')[2]: vpreds[k] for k in vpreds}

        # compute GAE(lambda), TD(lambda) estimators.
        seg_len = algo_config.get('segment_length')
        lam = algo_config.get('gae_lambda')
        gamma = algo_config.get('discount_gamma')
        advantages = {
            k: tc.zeros(seg_len+1, dtype=tc.float32)
            for k in relevant_reward_keys
        }
        for k in relevant_reward_keys:
            for t in reversed(range(0, seg_len)):  # T-1, ..., 0
                r_t = rewards[k][t]
                V_t = vpreds[k][t]
                V_tp1 = vpreds[k][t+1]
                A_tp1 = advantages[k][t+1]
                delta_t = -V_t + r_t + (1. - dones[t]) * gamma[k] * V_tp1
                A_t = delta_t + (1. - dones[t]) * gamma[k] * lam[k] * A_tp1
                advantages[k][t] = A_t
        advantages = {k: advantages[k][0:-1] for k in advantages}
        vpreds = {k: vpreds[k][0:-1] for k in vpreds}
        td_lam_rets = {k: advantages[k] + vpreds[k] for k in advantages}
        observations = observations[0:-1]

        return {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'logprobs': logprobs,
            'vpreds': vpreds,
            'advantages': advantages,
            'td_lambda_returns': td_lam_rets
        }

    def _compute_losses(self, mb):
        raise NotImplementedError

    def training_loop(self):
        policy_net = self._learning_system.get('policy_net')
        value_net = self._learning_system.get('value_net')
        max_steps = self._config.get('algo').get('max_steps')

        while self._learning_system.get('global_step') < max_steps:
           # finish this

    def evaluation_loop(self):
        raise NotImplementedError
