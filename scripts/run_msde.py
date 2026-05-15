import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import numpy as np


@hydra.main(version_base=None, config_path="../configs", config_name="msde")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)

    msde = instantiate(cfg.msde)

    file_path = "data/" + cfg.pca.save_file + ".npy"
    X = np.load(file_path)

    print("Running MSDE on data.")
    _, X_traj, X_total_dist, X_feature_dist  = msde.fit(X)

    save_path = "data/" + cfg.data
    np.save(save_path + "_trajectories.npy", X_traj)
    np.save(save_path + "_total_dist.npy", X_total_dist)
    np.save(save_path + "_feature_dist.npy", X_feature_dist)
    print("Saved trajectories and distances.")


if __name__ == "__main__":
    main()