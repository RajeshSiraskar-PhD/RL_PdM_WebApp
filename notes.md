# Notes

1. Violations are reflecting Replacements - ensure that "violations" are truely violations and replacements are not getting counted as violations
2. Confirm: Wear margin is not large - and is as close to 0 as possible 


We are now going to build two more variants:
'REINFORCE-V2' and 'REINFORCE-V2 with Attention'

For these the environments itself are going to be different. They will include tool-wear. This is because the model trained here is going to suggest actions for different settings of machine and the manfacturers have told us tool-wear measurements are available in real time

make MT_Env_v2 - similar to MT_Env except that the observation will now include tool-wear. Also create the attention equivalent AM_Env_v2 sub-classed on MT_Env_v2.

Then add training of 'REINFORCE-V2' and 'REINFORCE-V2 with Attention'

