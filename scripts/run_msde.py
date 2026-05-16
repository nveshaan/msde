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

    file_path = "data/" + cfg.pca.save_file + ".npy"
    X = np.load(file_path)

    print("Running MSDE on data.")
    _, X_total_dist, X_feature_dist  = msde.fit(X)

    save_path = "data/" + cfg.pca.save_file + ".hdf5"
    with h5py.File(save_path, 'a') as f:
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