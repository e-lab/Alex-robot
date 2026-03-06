# Alex robot controllers


## locomotion

controller module: locomotion_controller.py. It provides:

  - TwistCommand dataclass
  - VelocityPolicyLocomotionController with:
      - from_checkpoint(...) to load runner/policy
      - set_command(...) / get_command(...)
      - clamp_command(...)
      - step_policy() to run obs -> policy -> env.step

 
  - Updated play_alex_room.py to use the new controller:
      - Viewer now takes loco_ctrl instead of raw policy.
      - Manual key logic now sets commands through loco_ctrl.
      - Policy stepping now calls loco_ctrl.step_policy().
      - Runner/policy loading in main() moved into controller via from_checkpoint(...).



## LLM brain

Controller llm_brain_controller.py can do:

  - Takes available macro actions (action -> description)
  - Captures Alex head RGB frames via callback
  - Calls OpenAI Responses API with text + image
  - Produces JSON decision: action, done, summary
  - Executes atomic actions step-by-step until done or max_steps

### Play integration

  - Updated play_alex_room.py

Added CLI params:

  - --brain-prompt "find the door"
  - --brain-model gpt-4.1-mini
  - --brain-max-steps 30
  - --brain-step-interval-s 1.5

Behavior:

  - If --brain-prompt is set, viewer enables brain mode.
  - Brain captures head RGB camera frames and selects macro actions from alex_action_set.py.
  - Brain executes actions through the same macro interface already used in play.
  - Manual arrow-key fallback is disabled while brain is active (brain owns control loop).

### Also used

  - Existing locomotion_controller.py
  - Existing macro set alex_action_set.py

### verbose:

  - Added --verbose to play_alex_room.py.
  - Passed verbose into llm_brain_controller.py.
  - In LLMBrainController, when verbose=True, it now prints for each LLM step:
      - full system prompt
      - full user prompt
      - image payload metadata (jpeg_bytes, base64_chars)
      - raw model response text


### Run example

```bash
  cd /Users/euge/Code/github/Alex-robot/demos/alex-room-explore
  OPENAI_API_KEY=... python play_alex_room.py --viewer native --brain-prompt "find the door"
```

### Notes

  - Requires OPENAI_API_KEY.
  - Head-turn macros are available to the planner, but in current viewer they remain planner-level placeholders (no direct
    neck joint actuation yet).