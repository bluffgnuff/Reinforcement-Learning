from gym import envs

all_envs = envs.registry.all()

env_ids = [env_spec.id for env_spec in all_envs]

for id in sorted(env_ids):
    print(id, "\n")
