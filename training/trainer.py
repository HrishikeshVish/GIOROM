"""
Custom Trainer and Callback for Training a Reduced Order Model (ROM) with Transformers.

This script defines:
1. `CustomProgressCallback`: A Hugging Face TrainerCallback that integrates a progress bar and periodic evaluation 
   for monitoring model performance.
2. `ROMTrainer`: A custom trainer subclassing Hugging Face's `Trainer`, implementing a custom loss function and 
   logging mechanism for training physics-informed models.

Classes:
- `CustomProgressCallback`: Handles progress updates and periodic evaluations.
- `ROMTrainer`: Implements a custom loss function and training loop for physics-based models.

Key Features:
- Custom loss function incorporating physics constraints.
- Periodic evaluation of one-step and rollout MSE.
- Dynamic progress updates using `tqdm`.

Usage:
Instantiate `ROMTrainer` with model, training arguments, dataloaders, and evaluation functions,
then call `train()` to start training.

Example:
```python
trainer = ROMTrainer(model, args, train_dataloader, eval_dataloader, rollout_dataset, 
                     noise, eval_interval, rollout_interval, metadata, oneStepMSE, rolloutMSE)
trainer.train()
```
"""

from transformers import Trainer, TrainingArguments
import torch
import torch.nn as nn

from transformers import TrainerCallback
from tqdm import tqdm

class CustomProgressCallback(TrainerCallback):
    """
    Custom progress bar callback for Hugging Face's Trainer.

    This callback integrates a `tqdm` progress bar to track training progress and periodically evaluates 
    the model using one-step and rollout MSE.

    Args:
        eval_interval (int): Interval (in steps) at which the model is evaluated.
        rollout_interval (int): Interval (in steps) at which rollout MSE is computed.
        simulator (torch.nn.Module): The model being trained.
        metadata (dict): Metadata containing dataset-specific information like normalization statistics.
        eval_dataloader (torch.utils.data.DataLoader): Dataloader for evaluation.
        rollout_dataset (torch.utils.data.Dataset): Dataset for rollout-based evaluation.
        noise (torch.Tensor or None): Noise tensor for perturbing evaluation inputs.
        oneStepMSE (callable): Function to compute one-step MSE.
        rolloutMSE (callable): Function to compute rollout MSE.

    Methods:
        on_train_begin(args, state, control, **kwargs): Initializes the progress bar at the start of training.
        on_step_end(args, state, control, **kwargs): Updates progress bar and triggers periodic evaluations.
        on_epoch_end(args, state, control, **kwargs): Marks the end of an epoch in the progress bar.
        on_train_end(args, state, control, **kwargs): Closes the progress bar when training ends.
    """

    def __init__(self, eval_interval, rollout_interval, simulator, metadata, eval_dataloader, 
                 rollout_dataset, noise, oneStepMSE, rolloutMSE):
        self.training_bar = None
        self.eval_interval = eval_interval
        self.rollout_interval = rollout_interval
        self.simulator = simulator
        self.metadata = metadata
        self.eval_dataloader = eval_dataloader
        self.rollout_dataset = rollout_dataset
        self.noise = noise
        self.oneStepMSE = oneStepMSE
        self.rolloutMSE = rolloutMSE

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.training_bar = tqdm(total=state.max_steps, dynamic_ncols=True, desc="Training")
        self.current_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        # Update progress bar with loss or other custom information
        if self.training_bar is not None:
            loss = state.log_history[0]['loss']
            avg_loss = state.log_history[0]['avg_loss']
            lr = state.log_history[0]['lr']
            self.training_bar.set_postfix({"loss": loss, "avg_loss": avg_loss, "lr": lr})
            self.training_bar.update(1)
        
        if(state.log_history[0]['step'] % self.eval_interval == 0 and state.log_history[0]['step'] > 0):
            tqdm.write("...Evaluating Model...")
            eval_loss, onestep_mse = self.oneStepMSE(self.simulator, self.eval_dataloader, self.metadata, self.noise)
            tqdm.write(f"\nEval: Loss: {eval_loss}, One Step MSE: {onestep_mse}")
        if(state.log_history[0]['step'] % self.rollout_interval == 0 and state.log_history[0]['step'] > 0):
            print("...Evaluating Rollout...")
            rollout_mse = self.rolloutMSE(self.simulator, self.rollout_dataset, self.noise)
            tqdm.write(f"\nRollout MSE: {rollout_mse}")
        

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.training_bar is not None:
            self.training_bar.set_postfix({"epoch": state.epoch})
            self.training_bar.update(0)  # No update, just a marker for epoch end

    def on_train_end(self, args, state, control, **kwargs):
        # Close the progress bar when training ends
        if self.training_bar:
            self.training_bar.close()




class ROMTrainer(Trainer):
    """
    Custom Trainer for Reduced Order Model (ROM) training with physics constraints.

    This subclass of Hugging Face's `Trainer` provides:
    - A custom loss function integrating physics constraints.
    - Periodic evaluation of one-step and rollout MSE.
    - Suppression of excessive logging.

    Args:
        model (torch.nn.Module): The model to be trained.
        args (TrainingArguments): Hugging Face training arguments.
        train_dataloader (torch.utils.data.DataLoader, optional): Dataloader for training.
        eval_dataloader (torch.utils.data.DataLoader, optional): Dataloader for evaluation.
        rollout_dataset (torch.utils.data.Dataset, optional): Dataset for rollout-based evaluation.
        noise (torch.Tensor or None): Noise tensor for perturbing evaluation inputs.
        eval_interval (int, optional): Interval (in steps) for one-step evaluation.
        rollout_interval (int, optional): Interval (in steps) for rollout evaluation.
        data_collator (callable, optional): Function to collate data batches.
        metadata (dict): Metadata containing dataset-specific information like normalization statistics.
        optimizers (tuple, optional): Tuple of (optimizer, learning rate scheduler).
        oneStepMSE (callable, optional): Function to compute one-step MSE.
        rolloutMSE (callable, optional): Function to compute rollout MSE.

    Methods:
        compute_loss(model, inputs, return_outputs=False): Computes loss based on physics constraints.
        log(logs, start_time=None): Custom logging method to suppress excessive printouts.
        training_step(model, inputs, num_items_in_batch=None): Defines a single forward + backward pass.
        get_train_dataloader(): Returns the custom training dataloader.
    """

    def __init__(self, model, args, train_dataloader=None, eval_dataloader=None, rollout_dataset=None, 
                 noise=None, eval_interval=None, rollout_interval=None, 
                 data_collator=None, metadata=None, optimizers=None, oneStepMSE=None,
                 rolloutMSE=None):
        super().__init__(model=model, args=args, data_collator=data_collator)
        self.loss_fn = nn.MSELoss()
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.rollout_dataset = rollout_dataset
        self.noise = noise
        self.eval_interval = eval_interval
        self.rollout_interval = rollout_interval
        self.oneStepMSE = oneStepMSE
        self.rolloutMSE = rolloutMSE
        self.metadata = metadata
        if(optimizers is not None):
            self.optimizer, self.lr_scheduler = optimizers
        # Register the custom progress bar callback
        progress_bar_callback = CustomProgressCallback(eval_interval, rollout_interval, model, metadata, eval_dataloader, 
                 rollout_dataset, noise, oneStepMSE, rolloutMSE)
        self.add_callback(progress_bar_callback)
        self.total_loss = 0
        self.batch_count = 0

    def compute_loss(self, model, inputs, return_outputs=False):
        """Defines the custom loss function"""
        data = inputs.to(self.args.device)
        pred = model(data)

        acceleration = pred * torch.sqrt(torch.tensor(self.metadata['acc_std']).cuda() ** 2 + self.noise ** 2) + torch.tensor(self.metadata["acc_mean"]).cuda()
        recent_position = data.position_seq[:, -1]
        recent_velocity = recent_position - data.position_seq[:, -2]
        new_velocity = recent_velocity + acceleration
        new_position = recent_position + new_velocity
        loss = self.loss_fn(pred, data.y) + 1e9*self.loss_fn(new_position, data.target_pos)

        return (loss, pred) if return_outputs else loss
    def log(self, logs: dict[str, float], start_time: float = None):
        # Suppress printing by intercepting log messages
        if self.state.log_history:
            self.state.log_history[-1].update(logs)  # Update last log entry
        else:
            self.state.log_history.append(logs)  # Initialize if empty

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Defines the forward + backward pass per batch"""
        model.train()
        inputs = inputs.to(self.args.device)

        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        
        # Update loss tracking
        self.total_loss += loss.item()
        self.batch_count += 1
        avg_loss = self.total_loss / self.batch_count

        # Update Hugging Face's internal log
        self.log({"loss": loss.item(), "avg_loss": avg_loss, "lr": self.optimizer.param_groups[0]["lr"], "step": self.state.global_step})
        
        
        return loss.detach()
    def get_train_dataloader(self):
        """Returns the custom train DataLoader"""
        
        return self.train_dataloader