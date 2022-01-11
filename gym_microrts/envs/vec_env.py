
import os
import json
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

import gym
import gym_microrts

import jpype
from jpype.imports import registerDomain
import jpype.imports
from jpype.types import JArray, JInt

class MicroRTSGridModeVecEnv:
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 150
    }
    """
    [[0]x_coordinate*y_coordinate(x*y), [1]a_t(6), [2]p_move(4), [3]p_harvest(4), 
    [4]p_return(4), [5]p_produce_direction(4), [6]p_produce_unit_type(z), 
    [7]x_coordinate*y_coordinate(x*y)]
    Create a baselines VecEnv environment from a gym3 environment.
    :param env: gym3 environment to adapt
    """

    def __init__(self,
        num_selfplay_envs,
        num_bot_envs,
        partial_obs=False,
        max_steps=2000,
        render_theme=2,
        frame_skip=0,
        ai2s=[],
        map_paths=["maps/10x10/basesTwoWorkers10x10.xml"],
        reward_weight=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 5.0])):

        self.num_selfplay_envs = num_selfplay_envs
        self.num_bot_envs = num_bot_envs
        self.num_envs = num_selfplay_envs + num_bot_envs
        assert self.num_bot_envs == len(ai2s), "for each environment, a microrts ai should be provided"
        self.partial_obs = partial_obs
        self.max_steps = max_steps
        self.render_theme = render_theme
        self.frame_skip = frame_skip
        self.ai2s = ai2s
        self.map_paths = map_paths
        if len(map_paths) == 1:
            self.map_paths = [map_paths[0] for _ in range(self.num_envs)]
        else:
            assert len(map_paths) == self.num_envs, "if multiple maps are provided, they should be provided for each environment"
        self.reward_weight = reward_weight

        # read map
        self.microrts_path = os.path.join(gym_microrts.__path__[0], 'microrts')
        root = ET.parse(os.path.join(self.microrts_path, self.map_paths[0])).getroot()
        self.height, self.width = int(root.get("height")), int(root.get("width"))

        # launch the JVM
        if not jpype._jpype.isStarted():
            registerDomain("ts", alias="tests")
            registerDomain("ai")
            jars = [
                "microrts.jar", "lib/bots/Coac.jar", "lib/bots/Droplet.jar", "lib/bots/GRojoA3N.jar",
                "lib/bots/Izanagi.jar", "lib/bots/MixedBot.jar", "lib/bots/TiamatBot.jar", "lib/bots/UMSBot.jar",
                "lib/bots/mayariBot.jar" # "MindSeal.jar"
            ]
            for jar in jars:
                jpype.addClassPath(os.path.join(self.microrts_path, jar))
            jpype.startJVM(convertStrings=False)

        # start microrts client
        from rts.units import UnitTypeTable
        self.real_utt = UnitTypeTable()
        from ai.rewardfunction import RewardFunctionInterface, WinLossRewardFunction, ResourceGatherRewardFunction, AttackRewardFunction, ProduceWorkerRewardFunction, ProduceBuildingRewardFunction, ProduceCombatUnitRewardFunction, CloserToEnemyBaseRewardFunction
        self.rfs = JArray(RewardFunctionInterface)([
            WinLossRewardFunction(), 
            ResourceGatherRewardFunction(),  
            ProduceWorkerRewardFunction(),
            ProduceBuildingRewardFunction(),
            AttackRewardFunction(),
            ProduceCombatUnitRewardFunction(),
            # CloserToEnemyBaseRewardFunction(),
        ])
        self.start_client()

        # computed properties
        # [num_planes_hp(5), num_planes_resources(5), num_planes_player(5), 
        # num_planes_unit_type(z), num_planes_unit_action(6)]

        self.num_planes = [5, 5, 3, len(self.utt['unitTypes'])+1, 6]
        if partial_obs:
            self.num_planes = [5, 5, 3, len(self.utt['unitTypes'])+1, 6, 2]
        self.observation_space = gym.spaces.Box(low=0.0,
            high=1.0,
            shape=(self.height, self.width,
                    sum(self.num_planes)),
                    dtype=np.int32)

        self.num_planes_len = len(self.num_planes)
        self.num_planes_prefix_sum = [0]
        for num_plane in self.num_planes:
            self.num_planes_prefix_sum.append(self.num_planes_prefix_sum[-1] + num_plane)

        self.action_space = gym.spaces.MultiDiscrete(np.array([[6, 4, 4, 4, 4, len(self.utt['unitTypes']), 7 * 7]] * self.height * self.width).flatten())
        self.action_plane_space = gym.spaces.MultiDiscrete([6, 4, 4, 4, 4, len(self.utt['unitTypes']), 7 * 7])
        self.source_unit_idxs = np.tile(np.arange(self.height*self.width), (self.num_envs,1))
        self.source_unit_idxs = self.source_unit_idxs.reshape((self.source_unit_idxs.shape + (1,)))
        
    def start_client(self):

        from ts import JNIGridnetVecClient as Client
        from ai.core import AI
        self.vec_client = Client(
            self.num_selfplay_envs,
            self.num_bot_envs,
            self.max_steps,
            self.rfs,
            os.path.expanduser(self.microrts_path),
            self.map_paths,
            JArray(AI)([ai2(self.real_utt) for ai2 in self.ai2s]),
            self.real_utt,
            self.partial_obs,
        )
        self.render_client = self.vec_client.selfPlayClients[0] if len(self.vec_client.selfPlayClients) > 0 else self.vec_client.clients[0]
        # get the unit type table
        self.utt = json.loads(str(self.render_client.sendUTT()))

    def reset(self):
        responses = self.vec_client.reset([0]*self.num_envs)
        obs = [self._encode_obs(np.array(ro)) for ro in responses.observation]
        return np.array(obs)

    def _encode_obs(self, obs):
        obs = obs.reshape(len(obs), -1).clip(0, np.array([self.num_planes]).T-1)
        obs_planes = np.zeros((self.height * self.width, self.num_planes_prefix_sum[-1]), dtype=np.int32)
        obs_planes_idx = np.arange(len(obs_planes))
        obs_planes[obs_planes_idx,obs[0]] = 1

        for i in range(1, self.num_planes_len):
            obs_planes[obs_planes_idx,obs[i]+self.num_planes_prefix_sum[i]] = 1
        return obs_planes.reshape(self.height, self.width, -1)

    def step_async(self, actions):
        actions = actions.reshape((self.num_envs, self.width*self.height, -1))
        actions = np.concatenate((self.source_unit_idxs, actions), 2) # specify source unit
        actions = actions[np.where(self.source_unit_mask==1)] # valid actions
        action_counts_per_env = self.source_unit_mask.sum(1)
        java_actions = [None]*len(action_counts_per_env)
        action_idx = 0
        for outer_idx, action_count in enumerate(action_counts_per_env):
            java_valid_action = [None]*action_count
            for idx in range(action_count):
                java_valid_action[idx] = JArray(JInt)(actions[action_idx])
                action_idx += 1
            java_actions[outer_idx] = JArray(JArray(JInt))(java_valid_action)
        self.actions = JArray(JArray(JArray(JInt)))(java_actions)

    def step_wait(self):
        responses = self.vec_client.gameStep(self.actions, [0]*self.num_envs)
        reward, done = np.array(responses.reward), np.array(responses.done)
        obs = [self._encode_obs(np.array(ro)) for ro in responses.observation]
        infos = [{"raw_rewards": item} for item in reward]
        return np.array(obs), reward @ self.reward_weight, done[:,0], infos

    def step(self, ac):
        self.step_async(ac)
        return self.step_wait()

    def getattr_depth_check(self, name, already_found):
        """Check if an attribute reference is being hidden in a recursive call to __getattr__
        :param name: (str) name of attribute to check for
        :param already_found: (bool) whether this attribute has already been found in a wrapper
        :return: (str or None) name of module whose attribute is being shadowed, if any.
        """
        if hasattr(self, name) and already_found:
            return "{0}.{1}".format(type(self).__module__, type(self).__name__)
        else:
            return None

    def render(self, mode="human"):
        if mode == "human":
            self.render_client.render(False)
        elif mode == 'rgb_array':
            bytes_array = np.array(self.render_client.render(True))
            image = Image.frombytes("RGB", (640, 640), bytes_array)
            return np.array(image)[:,:,::-1]

    def close(self):
        if jpype._jpype.isStarted():
            self.vec_client.close()
            jpype.shutdownJVM()

    def get_action_mask(self):
        action_mask = np.array(self.vec_client.getMasks(0))
        self.source_unit_mask = action_mask[:,:,:,0].reshape(self.num_envs, -1)
        action_type_and_parameter_mask = action_mask[:,:,:,1:].reshape(self.num_envs, self.height*self.width, -1)
        return action_type_and_parameter_mask

class MicroRTSBotVecEnv(MicroRTSGridModeVecEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 150
    }

    def __init__(self,
        ai1s=[],
        ai2s=[],
        partial_obs=False,
        max_steps=2000,
        render_theme=2,
        map_paths="maps/10x10/basesTwoWorkers10x10.xml",
        reward_weight=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 5.0])):

        self.ai1s = ai1s
        self.ai2s = ai2s
        assert len(ai1s) == len(ai2s), "for each environment, a microrts ai should be provided"
        self.num_envs = len(ai1s)
        self.partial_obs = partial_obs
        self.max_steps = max_steps
        self.render_theme = render_theme
        self.map_paths = map_paths
        self.reward_weight = reward_weight

        # read map
        self.microrts_path = os.path.join(gym_microrts.__path__[0], 'microrts')
        root = ET.parse(os.path.join(self.microrts_path, self.map_paths[0])).getroot()
        self.height, self.width = int(root.get("height")), int(root.get("width"))

        # launch the JVM
        if not jpype._jpype.isStarted():
            registerDomain("ts", alias="tests")
            registerDomain("ai")
            jars = [
                "microrts.jar", "lib/bots/Coac.jar", "lib/bots/Droplet.jar", "lib/bots/GRojoA3N.jar",
                "lib/bots/Izanagi.jar", "lib/bots/MixedBot.jar", "lib/bots/TiamatBot.jar", "lib/bots/UMSBot.jar",
                "lib/bots/mayariBot.jar" # "MindSeal.jar"
            ]
            for jar in jars:
                jpype.addClassPath(os.path.join(self.microrts_path, jar))
            jpype.startJVM(convertStrings=False)

        # start microrts client
        from rts.units import UnitTypeTable
        self.real_utt = UnitTypeTable()
        from ai.rewardfunction import RewardFunctionInterface, WinLossRewardFunction, ResourceGatherRewardFunction, AttackRewardFunction, ProduceWorkerRewardFunction, ProduceBuildingRewardFunction, ProduceCombatUnitRewardFunction, CloserToEnemyBaseRewardFunction
        self.rfs = JArray(RewardFunctionInterface)([
            WinLossRewardFunction(), 
            ResourceGatherRewardFunction(),  
            ProduceWorkerRewardFunction(),
            ProduceBuildingRewardFunction(),
            AttackRewardFunction(),
            ProduceCombatUnitRewardFunction(),
            # CloserToEnemyBaseRewardFunction(),
        ])
        self.start_client()

        # computed properties
        # [num_planes_hp(5), num_planes_resources(5), num_planes_player(5), 
        # num_planes_unit_type(z), num_planes_unit_action(6)]

        self.num_planes = [5, 5, 3, len(self.utt['unitTypes'])+1, 6]
        if partial_obs:
            self.num_planes = [5, 5, 3, len(self.utt['unitTypes'])+1, 6, 2]
        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Discrete(2)

    def start_client(self):

        from ts import JNIGridnetVecClient as Client
        from ai.core import AI
        self.vec_client = Client(
            self.max_steps,
            self.rfs,
            os.path.expanduser(self.microrts_path),
            self.map_paths,
            JArray(AI)([ai1(self.real_utt) for ai1 in self.ai1s]),
            JArray(AI)([ai2(self.real_utt) for ai2 in self.ai2s]),
            self.real_utt,
            self.partial_obs,
        )
        self.render_client = self.vec_client.botClients[0]
        # get the unit type table
        self.utt = json.loads(str(self.render_client.sendUTT()))

    def reset(self):
        responses = self.vec_client.reset([0 for _ in range(self.num_envs)])
        raw_obs, reward, done, info = np.ones((self.num_envs,2)), np.array(responses.reward), np.array(responses.done), {}
        return raw_obs

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        responses = self.vec_client.gameStep(self.actions, [0 for _ in range(self.num_envs)])
        raw_obs, reward, done = np.ones((self.num_envs,2)), np.array(responses.reward), np.array(responses.done)
        infos = [{"raw_rewards": item} for item in reward]
        return raw_obs, reward @ self.reward_weight, done[:,0], infos

    def step(self, ac):
        self.step_async(ac)
        return self.step_wait()

    def getattr_depth_check(self, name, already_found):
        """Check if an attribute reference is being hidden in a recursive call to __getattr__
        :param name: (str) name of attribute to check for
        :param already_found: (bool) whether this attribute has already been found in a wrapper
        :return: (str or None) name of module whose attribute is being shadowed, if any.
        """
        if hasattr(self, name) and already_found:
            return "{0}.{1}".format(type(self).__module__, type(self).__name__)
        else:
            return None

    def render(self, mode="human"):
        if mode == "human":
            self.render_client.render(False)
        elif mode == 'rgb_array':
            bytes_array = np.array(self.render_client.render(True))
            image = Image.frombytes("RGB", (640, 640), bytes_array)
            return np.array(image)[:,:,::-1]

    def close(self):
        if jpype._jpype.isStarted():
            self.vec_client.close()
            jpype.shutdownJVM()


class MicroRTSGridModeSharedMemVecEnv(MicroRTSGridModeVecEnv):
    """
    Similar function to `MicroRTSGridModeVecEnv` but uses shared mem buffers for
    zero-copy data exchange between NumPy and JVM runtimes. Drastically improves
    performance of the environment with some limitations introduced to the API.
    Notably, all games should be performed on the same map.
    """

    def __init__(
        self,
        num_selfplay_envs,
        num_bot_envs,
        partial_obs=False,
        max_steps=2000,
        render_theme=2,
        frame_skip=0,
        ai2s=[],
        map_paths=["maps/10x10/basesTwoWorkers10x10.xml"],
        reward_weight=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 5.0])
    ):
        if len(map_paths) > 1 and len(set(map_paths)) > 1:
            raise ValueError("Mem shared environment requires all games to be played on the same map.")

        super(MicroRTSGridModeSharedMemVecEnv, self).__init__(
            num_selfplay_envs,
            num_bot_envs,
            partial_obs,
            max_steps,
            render_theme,
            frame_skip,
            ai2s,
            map_paths,
            reward_weight,
        )

    def _allocate_shared_buffer(self, nbytes):
        from java.nio import ByteOrder
        from jpype.nio import convertToDirectBuffer

        c_buffer = bytearray(nbytes)
        jvm_buffer = convertToDirectBuffer(c_buffer).order(ByteOrder.nativeOrder()).asIntBuffer()
        np_buffer = np.asarray(jvm_buffer, order="C")
        return jvm_buffer, np_buffer

    def start_client(self):

        from ts import JNIGridnetSharedMemVecClient as Client
        from ai.core import AI

        # xxx(okachaiev): there's a race condition here...
        # in order to start client, I need to pre-allocate buffers
        # to pre-allocate buffers I need to know dims of all array,
        # which will be determined by UTT requested from the running client
        # need to introduce new API for getting this information prior to
        # running the client
        self.num_feature_planes = 27
        self.masks_dim = 78

        # pre-allocate shared buffers with JVM
        obs_nbytes = self.num_envs * self.height * self.width * self.num_feature_planes * 4
        obs_jvm_buffer, obs_np_buffer = self._allocate_shared_buffer(obs_nbytes)
        self.obs = obs_np_buffer.reshape((self.num_envs, self.height, self.width, self.num_feature_planes))

        unit_mask_nbytes = self.num_envs * self.height * self.width * 4
        unit_mask_jvm_buffer, unit_mask_np_buffer = self._allocate_shared_buffer(unit_mask_nbytes)
        self.source_unit_mask = unit_mask_np_buffer.reshape((self.num_envs, self.height*self.width))

        action_mask_nbytes = self.num_envs * self.height * self.width * self.masks_dim * 4
        action_mask_jvm_buffer, action_mask_np_buffer = self._allocate_shared_buffer(action_mask_nbytes)
        self.action_mask = action_mask_np_buffer.reshape((self.num_envs, self.height*self.width, self.masks_dim))

        self.vec_client = Client(
            self.num_selfplay_envs,
            self.num_bot_envs,
            self.max_steps,
            self.rfs,
            os.path.expanduser(self.microrts_path),
            self.map_paths[0],
            JArray(AI)([ai2(self.real_utt) for ai2 in self.ai2s]),
            self.real_utt,
            self.partial_obs,
            obs_jvm_buffer,
            unit_mask_jvm_buffer,
            action_mask_jvm_buffer,
        )
        self.render_client = self.vec_client.selfPlayClients[0] if len(self.vec_client.selfPlayClients) > 0 else self.vec_client.clients[0]
        # get the unit type table
        self.utt = json.loads(str(self.render_client.sendUTT()))

    def reset(self):
        self.vec_client.reset([0]*self.num_envs)
        return self.obs

    def step_wait(self):
        responses = self.vec_client.gameStep(self.actions, [0]*self.num_envs)
        reward, done = np.array(responses.reward), np.array(responses.done)
        infos = [{"raw_rewards": item} for item in reward]
        return self.obs, reward @ self.reward_weight, done[:,0], infos

    def get_action_mask(self):
        self.vec_client.getMasks(0)
        return self.action_mask
