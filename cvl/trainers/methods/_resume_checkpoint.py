

def resume_classification_checkpoint(self, **kwargs):
    raise NotImplemented(f"Can not implement this because I don't know how to resume TensorBoard logger")

    params = utils.read_pickle_file(params_path)

    self.max_epochs = params["max_epochs"]
    self.checkpoints_path = params["checkpoints_path"]
    self.current_loss = params["current_loss"]
    self.prev_loss = params["prev_loss"]
    self.dataloaders = params["dataloaders"]
    self.model = params["model"]
    self.optimizer = params["optimizer"]
    self.scheduler = params["scheduler"]
    self.evaluators = params["evaluators"]
    self.criterion = params["criterion"]

    print(f"{param_path} Loaded")

    sd = torch.load(f=checkpoint_path, map_location='cpu', weights_only=True)
    epoch = sd["epoch"]
    self.model.load_state_dict(sd["model_state_dict"], strict=True)
    self.optimizer.load_state_dict(sd["optimizer_state_dict"], strict=True)

    print(f"{checkpoint_path} Loaded")

    self.fit(epoch + 1)