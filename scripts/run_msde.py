import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import numpy as np
import h5py
import os


@hydra.main(version_base=None, config_path="../configs", config_name="msde")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)

    msde = instantiate(cfg.msde)

    path = str(cfg.pca.path) + ".npy"
    if not os.path.exists(path):
        raise FileExistsError(f"{path} does not exist. Please fit PCA.")
    X = np.load(path)

    print("Running MSDE on data.")
    _, X_total_dist, X_feature_dist  = msde.fit(X)

    path = str(cfg.path)
    with h5py.File(path, 'a') as f:
        for key in ["total_dist", "feature_dist", "trajectory"]:
            if key in f:
                del f[key]
                
        f["total_dist"] = X_total_dist
        f["feature_dist"] = X_feature_dist
        
        traj_group = f.create_group("trajectory")
        
        for i in range(cfg.msde.max_iters_shift+1):
            traj_file = f"temp/traj_{i}.npy"
            if os.path.exists(traj_file):
                arr = np.load(traj_file)
                traj_group.create_dataset(str(i), data=arr, compression="gzip", compression_opts=4)
                os.remove(traj_file)
            else:
                break

    print("Saved trajectories and distances.")


if __name__ == "__main__":
    main()