# pytorch_drl
Deep reinforcement learning algorithms implemented in Pytorch. 

In progress...

Config usage: 
style this section after pytorch_ddp_resnet.
Mention that we support multiple intrinsic rewards, including their combination. In general, we follow Burda et al., 2018 and Badia et al., 2020 in predicting these reward streams separately, and this is reflected in the naming conventions used in our config files. For value prediction heads, we use 'value_{reward_name}' as the format for each value head. Likewise, for action value heads, we use 'action_value_{reward_name}'. The default reward name is 'extrinsic'. We also plan on supporting multiple auxiliary losses, but we leave this for later work. 