import wandb

run = wandb.init(project="runs-from-for-loop", id='test')
for x in range(10):
    for y in range (100):
        wandb.log({"metric": x+y})
run.finish()

