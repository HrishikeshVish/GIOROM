from transformers import TrainingArguments

def train(params, optimizer, scheduler, HFTrainer, simulator, train_loader, eval_loader=None, 
          metadata=None, rollout_dataset=None, oneStepMSE=None, rolloutMSE=None,
          output_dir='saved_models', log_dir='logs'):
    ############################################################################
    """Train the model with the given parameters and data loaders."""
    # Convert YAML config to Hugging Face TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=params.epoch,
        per_device_train_batch_size=params.batch_size,
        learning_rate=params.lr,
        weight_decay=params.weight_decay,
        logging_dir=log_dir,
        logging_steps=params.log_interval,
        save_steps=params.save_interval,
        disable_tqdm=True
    )

    # Initialize model
    model = simulator

    # Use custom trainer
    trainer = HFTrainer(
        model=model,
        args=training_args,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        rollout_dataset=rollout_dataset,
        noise=params.noise,
        eval_interval = params.eval_interval,
        rollout_interval = params.rollout_interval,
        oneStepMSE=oneStepMSE,
        rolloutMSE=rolloutMSE,
        metadata = metadata,
        optimizers=(optimizer, scheduler)
    )

    # Train the model
    trainer.train()