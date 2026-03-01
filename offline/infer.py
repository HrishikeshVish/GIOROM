from models.giorom3d_T import PhysicsEngine
from models.config import TimeStepperConfig

import os
time_stepper_config = TimeStepperConfig()

simulator = PhysicsEngine(time_stepper_config)
repo_id = "hrishivish23/giorom-3d-t-sand3d-long"
time_stepper_config = time_stepper_config.from_pretrained(repo_id)
simulator = simulator.from_pretrained(repo_id, config=time_stepper_config)

