import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs


from sklearn.metrics import average_precision_score

from rdkit.DataStructs.cDataStructs import UIntSparseIntVect, ExplicitBitVect


from typing import Union, Tuple, List

import numpy as np

import argparse


def smiles_to_fingerprints(
    smiles: Union[List[str], str],
    radius: int = 2,
    n_bits: int = 2048,
    use_counts: bool = True,
    use_features: bool = True,
) -> Tuple[List[Union[UIntSparseIntVect, ExplicitBitVect]], np.ndarray, List[int]]:
    """Generates Morgan-like fingerprints of smile strings from a list or
    from a single SMILES string by using RDKit. Returns index of valid SMILES.

    Args:
        smiles (Union[List[str], str]): List of SMILES strings to convert
        radius (int): Radius of fingerprint. Default to 2.
        n_bits (int): Number of bits to use in feature vector. Defaults to 2048.
        use_counts (bool, optional): Whether to use count-based feature vectors or not. Defaults to True.
        use_features (bool, optional): Whether to use feature-based invariants or not. Defaults to True.

    Returns:
        Tuple[List[Union[UIntSparseIntVect,ExplicitBitVect]],np.ndarray,List[int]]: Generated fingerprints (both in RDKit format and as numpy ndarray) and indices of valid SMILES
    """

    fps = []
    fps_array = []
    valid_indices = []

    if isinstance(smiles, str):
        smiles = [smiles]

    # If count-based feature vectors as used
    if use_counts:
        for i, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi, sanitize=True)
            # Check if generated SMILES gives chemically feasible compound according to RDKit
            if mol is not None:
                valid_indices.append(i)
                fp = AllChem.GetHashedMorganFingerprint(
                    mol,
                    radius=radius,
                    nBits=n_bits,
                    useFeatures=use_features,
                )
                fps.append(fp)
                array = np.zeros((0,), dtype=np.float32)
                DataStructs.ConvertToNumpyArray(fp, array)
                fps_array.append(array)
    # If binary feature vectors are used
    else:
        for i, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi, sanitize=True)
            # Check if generated SMILES gives chemically feasible compound according to RDKit
            if mol is not None:
                valid_indices.append(i)
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol,
                    radius=radius,
                    nBits=n_bits,
                )
                fps.append(fp)
                array = np.zeros((0,), dtype=np.float32)
                DataStructs.ConvertToNumpyArray(fp, array)
                fps_array.append(array)

    if len(fps) == 0:
        return None
    elif len(fps) == 1:
        return fps[0], fps_array[0], valid_indices[0]

    return fps, np.array(fps_array), valid_indices


def generate_fps_and_activity(fp_radius=2, fp_bits=2048, fp_counts=True):

    df = pd.read_csv("predictive_models/DRD2/DRD2.csv", sep=";")[
        ["Activity_Flag", "SMILES"]
    ]

    smiles = df["SMILES"].copy().tolist()

    df_activity = df["Activity_Flag"].copy()

    df_activity.replace("N", 0.0, inplace=True)
    df_activity.replace("A", 1.0, inplace=True)

    fps, fps_array, valid_idx = smiles_to_fingerprints(
        smiles,
        radius=fp_radius,
        n_bits=fp_bits,
        use_counts=fp_counts,
        use_features=True,
    )

    print(f"# invalid molecules")
    # Activity
    df_activity = df_activity.iloc[valid_idx]
    df_activity.reset_index(drop=True, inplace=True)
    activity = df_activity.to_numpy()

    np.save(
        f"predictive_models/DRD2/DRD2_activity.npy",
        activity,
    )

    if fp_counts:

        with open(
            f"predictive_models/DRD2/DRD2_ecfp{int(2*fp_radius)}c.pkl", "wb"
        ) as file_path:
            pickle.dump(fps, file_path)
        np.save(
            f"predictive_models/DRD2/DRD2_ecfp{int(2*fp_radius)}c.npy",
            fps_array,
        )
        np.save(
            f"predictive_models/DRD2/valid_idx_DRD2_ecfp{int(2*fp_radius)}c.npy",
            valid_idx,
        )

    else:
        with open(
            f"predictive_models/DRD2/DRD2_ecfp{int(2*fp_radius)}.pkl", "wb"
        ) as file_path:
            pickle.dump(fps, file_path)

        np.save(
            f"predictive_models/DRD2/DRD2_ecfp{int(2*fp_radius)}.npy",
            fps_array,
        )

        np.save(
            f"predictive_models/valid_idx_DRD2_ecfp{int(2*fp_radius)}.npy",
            valid_idx,
        )

    return fps, fps_array, activity, valid_idx


def create_RF_model(fps: np.ndarray, activity: np.ndarray) -> object:
    """Train random forest activity model

    Args:
        fps (np.ndarray): Fingerprints [n_fps,n_bits]
        activity (np.ndarray): Activity of each fingerprint [n_fps,]

    Returns:
        object: Trained scikit-learn model
    """

    rfc = RandomForestClassifier(
        n_estimators=1300,
        criterion="gini",
        max_depth=300,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=42,
        verbose=1,
        warm_start=False,
        class_weight="balanced",
        ccp_alpha=0.0,
        max_samples=None,
    )

    rfc.fit(fps, activity)

    probs = rfc.predict_proba(fps)

    average_precision = average_precision_score(activity, probs[:, 1], pos_label=1)
    print(f"Train average_precision: {average_precision}", flush=True)

    return rfc


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--fp_radius", type=int, default=2)
    parser.add_argument("--fp_bits", type=int, default=2048)
    parser.add_argument("--fp_counts", action="store_true")
    parser.add_argument("--generate_fps", action="store_true")

    args = parser.parse_args()
    print(args, flush=True)

    # Generate new fingerprints if input argument is given
    if args.generate_fps:
        print("Generating fingerprints")
        fps, fps_array, activity, valid_idx = generate_fps_and_activity(
            args.fp_radius, args.fp_bits, args.fp_counts
        )
    else:
        activity = np.load(f"predictive_models/DRD2/DRD2_activity.npy")
        if args.fp_counts:

            fps_array = np.load(
                f"predictive_models/DRD2/DRD2_ecfp{int(2*args.fp_radius)}c.npy"
            )

        else:

            fps_array = np.load(
                f"predictive_models/DRD2/DRD2_ecfp{int(2*args.fp_radius)}.npy"
            )

    print("Creating RF model...", flush=True)

    print(f"fps shape {fps_array.shape}", flush=True)
    print(f"Activity shape {activity.shape}", flush=True)

    rf_cls = create_RF_model(fps_array, activity)

    print("Done.")

    if args.fp_counts:
        with open(
            f"predictive_models/DRD2/RF_DRD2_ecfp{int(2*args.fp_radius)}c.pkl",
            "wb",
        ) as f:
            pickle.dump(rf_cls, f)
    else:
        with open(
            f"predictive_models/DRD2/RF_DRD2_ecfp{int(2*args.fp_radius)}.pkl",
            "wb",
        ) as f:
            pickle.dump(rf_cls, f)
