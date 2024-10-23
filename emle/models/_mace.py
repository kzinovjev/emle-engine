#######################################################################
# EMLE-Engine: https://github.com/chemle/emle-engine
#
# Copyright: 2023-2024
#
# Authors: Lester Hedges   <lester.hedges@gmail.com>
#          Kirill Zinovjev <kzinovjev@gmail.com>
#
# EMLE-Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# EMLE-Engine is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EMLE-Engine. If not, see <http://www.gnu.org/licenses/>.
#####################################################################

"""MACEEMLE model implementation."""

__author__ = "Joao Morado"
__email__ = "joaomorado@gmail.com>"

__all__ = ["MACEEMLE"]

import os as _os
import torch as _torch

from typing import List

from ._emle import EMLE as _EMLE
from ._emle import _has_nnpops
from ._utils import _get_neighbor_pairs

try:
    from mace.calculators.foundations_models import mace_off as _mace_off

    _has_mace = True
except:
    _has_mace = False

try:
    from e3nn.util import jit as _e3nn_jit

    _has_e3nn = True
except:
    _has_e3nn = False


class MACEEMLE(_torch.nn.Module):

    # Class attributes.

    # A flag for type inference. TorchScript doesn't support inheritance, so
    # we need to check for an object of type torch.nn.Module, and that it has
    # the required _is_emle attribute.
    _is_emle = True

    def __init__(
        self,
        emle_model=None,
        emle_method="electrostatic",
        alpha_mode="species",
        mm_charges=None,
        mace_model=None,
        atomic_numbers=None,
        device=None,
        dtype=None,
    ):
        """
        Constructor.

        Parameters
        ----------

        emle_model: str
            Path to a custom EMLE model parameter file. If None, then the
            default model for the specified 'alpha_mode' will be used.

        emle_method: str
            The desired embedding method. Options are:
                "electrostatic":
                    Full ML electrostatic embedding.
                "mechanical":
                    ML predicted charges for the core, but zero valence charge.
                "nonpol":
                    Non-polarisable ML embedding. Here the induced component of
                    the potential is zeroed.
                "mm":
                    MM charges are used for the core charge and valence charges
                    are set to zero.

        alpha_mode: str
            How atomic polarizabilities are calculated.
                "species":
                    one volume scaling factor is used for each species
                "reference":
                    scaling factors are obtained with GPR using the values learned
                    for each reference environmentw

        mm_charges: List[float], Tuple[Float], numpy.ndarray, torch.Tensor
            List of MM charges for atoms in the QM region in units of mod
            electron charge. This is required if the 'mm' method is specified.

        mace_model: str
            Name of the MACE-OFF23 models to use.
            Available models are 'mace-off23-small', 'mace-off23-medium', 'mace-off23-large'.
            To use a locally trained MACE model, provide the path to the model file.
            If None, the MACE-OFF23(S) model will be used by default.

        atomic_numbers: List[int], Tuple[int], numpy.ndarray, torch.Tensor (N_ATOMS,)
            List of atomic numbers to use in the MACE model.

        device: torch.device
            The device on which to run the model.

        dtype: torch.dtype
            The data type to use for the models floating point tensors.
        """

        # Call the base class constructor.
        super().__init__()

        if not _has_mace:
            raise ImportError(
                'mace is required to use the MACEEMLE model. Install it with "pip install mace-torch"'
            )
        if not _has_e3nn:
            raise ImportError("e3nn is required to compile the MACEmodel.")
        if not _has_nnpops:
            raise ImportError("NNPOps is required to use the MACEEMLE model.")

        if device is not None:
            if not isinstance(device, _torch.device):
                raise TypeError("'device' must be of type 'torch.device'")
        else:
            device = _torch.get_default_device()

        if dtype is not None:
            if not isinstance(dtype, _torch.dtype):
                raise TypeError("'dtype' must be of type 'torch.dtype'")
        else:
            self._dtype = _torch.get_default_dtype()

        if atomic_numbers is not None:
            if isinstance(atomic_numbers, _np.ndarray):
                atomic_numbers = atomic_numbers.tolist()
            if isinstance(atomic_numbers, (list, tuple)):
                if not all(isinstance(i, int) for i in atomic_numbers):
                    raise ValueError("'atomic_numbers' must be a list of integers")
                else:
                    atomic_numbers = _torch.tensor(atomic_numbers, dtype=_torch.int64)
            if not isinstance(atomic_numbers, _torch.Tensor):
                raise TypeError("'atomic_numbers' must be of type 'torch.Tensor'")
            # Check that they are integers.
            if atomic_numbers.dtype != _torch.int64:
                raise ValueError("'atomic_numbers' must be of dtype 'torch.int64'")
            self.register_buffer("_atomic_numbers", atomic_numbers)
        else:
            self.register_buffer(
                "_atomic_numbers",
                _torch.tensor([], dtype=_torch.int64, requires_grad=False),
            )

        # Create an instance of the EMLE model.
        self._emle = _EMLE(
            model=emle_model,
            method=emle_method,
            alpha_mode=alpha_mode,
            atomic_numbers=(atomic_numbers if atomic_numbers is not None else None),
            mm_charges=mm_charges,
            device=device,
            dtype=dtype,
            create_aev_calculator=True,
        )

        # Load the MACE model.
        if mace_model is not None:
            if not isinstance(mace_model, str):
                raise TypeError("'mace_model' must be of type 'str'")
            # Convert to lower case and remove whitespace.
            mace_model = mace_model.lower().replace(" ", "")
            if mace_model.startswith("mace-off23"):
                size = mace_model.split("-")[-1]
                if not size in ["small", "medium", "large"]:
                    raise ValueError(
                        f"Unsupported MACE model: '{mace_model}'. Available MACE-OFF23 models are "
                        "'mace-off23-small', 'mace-off23-medium', 'mace-off23-large'"
                    )
                self._mace = _mace_off(model=size, device=device, return_raw_model=True)
            else:
                # Assuming that the model is a local model.
                if _os.path.exists(mace_model):
                    self._mace = _torch.load(mace_model, map_location=device)
                else:
                    raise FileNotFoundError(f"MACE model file not found: {mace_model}")
        else:
            # If no MACE model is provided, use the default MACE-OFF23(S) model.
            self._mace = _mace_off(model="small", device=device, return_raw_model=True)

        # Compile the model.
        self._mace = _e3nn_jit.compile(self._mace).to(self._dtype)

        # Create the z_table of the MACE model.
        self._z_table = [int(z.item()) for z in self._mace.atomic_numbers]

        if len(self._atomic_numbers) > 0:
            # Get the node attributes.
            node_attrs = self._get_node_attrs(self._atomic_numbers)
            self.register_buffer("_node_attrs", node_attrs.to(self._dtype))
            self.register_buffer(
                "_ptr",
                _torch.tensor(
                    [0, node_attrs.shape[0]], dtype=_torch.long, requires_grad=False
                ),
            )
            self.register_buffer(
                "_batch",
                _torch.zeros(
                    node_attrs.shape[0], dtype=_torch.long, requires_grad=False
                ),
            )
        else:
            # Initialise the node attributes.
            self.register_buffer("_node_attrs", _torch.tensor([], dtype=self._dtype))
            self.register_buffer(
                "_ptr", _torch.tensor([], dtype=_torch.long, requires_grad=False)
            )
            self.register_buffer(
                "_batch", _torch.tensor([], dtype=_torch.long, requires_grad=False)
            )

        # No PBCs for now.
        self.register_buffer(
            "_pbc",
            _torch.tensor(
                [False, False, False], dtype=_torch.bool, requires_grad=False
            ),
        )
        self.register_buffer(
            "_cell", _torch.zeros((3, 3), dtype=self._dtype, requires_grad=False)
        )

        # Set the _get_neighbor_pairs method on the instance.
        self._get_neighbor_pairs = _get_neighbor_pairs

    @staticmethod
    def _to_one_hot(indices: _torch.Tensor, num_classes: int) -> _torch.Tensor:
        """
        Convert a tensor of indices to one-hot encoding.

        Parameters
        ----------

        indices: torch.Tensor
            Tensor of indices.

        num_classes: int
            Number of classes of atomic numbers.

        Returns
        -------

        oh: torch.Tensor
            One-hot encoding of the indices.
        """
        shape = indices.shape[:-1] + (num_classes,)
        oh = _torch.zeros(shape, device=indices.device).view(shape)
        return oh.scatter_(dim=-1, index=indices, value=1)

    @staticmethod
    def _atomic_numbers_to_indices(
        atomic_numbers: _torch.Tensor, z_table: List[int]
    ) -> _torch.Tensor:
        """
        Get the indices of the atomic numbers in the z_table.

        Parameters
        ----------

        atomic_numbers: torch.Tensor (N_ATOMS,)
            Atomic numbers of QM atoms.

        z_table: List[int]
            List of atomic numbers in the MACE model.

        Returns
        -------

        indices: torch.Tensor (N_ATOMS, 1)
            Indices of the atomic numbers in the z_table.
        """
        return _torch.tensor(
            [z_table.index(z) for z in atomic_numbers], dtype=_torch.long
        ).unsqueeze(-1)

    def _get_node_attrs(self, atomic_numbers: _torch.Tensor) -> _torch.Tensor:
        """
        Internal method to get the node attributes for the MACE model.

        Parameters
        ----------

        atomic_numbers: torch.Tensor (N_ATOMS,)
            Atomic numbers of QM atoms.

        Returns
        -------

        node_attrs: torch.Tensor (N_ATOMS, N_FEATURES)
            Node attributes for the MACE model.
        """
        ids = self._atomic_numbers_to_indices(atomic_numbers, z_table=self._z_table)
        return self._to_one_hot(ids, num_classes=len(self._z_table))

    def to(self, *args, **kwargs):
        """
        Performs Tensor dtype and/or device conversion on the model.
        """
        self._emle = self._emle.to(*args, **kwargs)
        self._mace = self._mace.to(*args, **kwargs)
        return self

    def cpu(self, **kwargs):
        """
        Move all model parameters and buffers to CPU memory.
        """
        self._emle = self._emle.cpu(**kwargs)
        self._mace = self._mace.cpu(**kwargs)
        if self._atomic_numbers is not None:
            self._atomic_numbers = self._atomic_numbers.cpu(**kwargs)
        return self

    def cuda(self, **kwargs):
        """
        Move all model parameters and buffers to CUDA memory.
        """
        self._emle = self._emle.cuda(**kwargs)
        self._mace = self._mace.cuda(**kwargs)
        if self._atomic_numbers is not None:
            self._atomic_numbers = self._atomic_numbers.cuda(**kwargs)
        return self

    def double(self):
        """
        Cast all floating point model parameters and buffers to float64 precision.
        """
        self._emle = self._emle.double()
        self._mace = self._mace.double()
        return self

    def float(self):
        """
        Cast all floating point model parameters and buffers to float32 precision.
        """
        self._emle = self._emle.float()
        self._mace = self._mace.float()
        return self

    def forward(self, atomic_numbers, charges_mm, xyz_qm, xyz_mm):
        """
        Compute the the MACE and static and induced EMLE energy components.

        Parameters
        ----------

        atomic_numbers: torch.Tensor (N_QM_ATOMS,)
            Atomic numbers of QM atoms.

        charges_mm: torch.Tensor (max_mm_atoms,)
            MM point charges in atomic units.

        xyz_qm: torch.Tensor (N_QM_ATOMS, 3)
            Positions of QM atoms in Angstrom.

        xyz_mm: torch.Tensor (N_MM_ATOMS, 3)
            Positions of MM atoms in Angstrom.

        Returns
        -------

        result: torch.Tensor (3,)
            The ANI2x and static and induced EMLE energy components in Hartree.
        """
        # Get the device.
        device = xyz_qm.device

        # Get the edge index and shifts for this configuration.
        edge_index, shifts = self._get_neighbor_pairs(
            xyz_qm, None, self._mace.r_max, self._dtype, device
        )

        if not _torch.equal(atomic_numbers, self._atomic_numbers):
            # Update the node attributes if the atomic numbers have changed.
            self._node_attrs = (
                self._get_node_attrs(atomic_numbers).to(self._dtype).to(device)
            )
            self._ptr = _torch.tensor(
                [0, self._node_attrs.shape[0]], dtype=_torch.long, requires_grad=False
            ).to(device)
            self._batch = _torch.zeros(self._node_attrs.shape[0], dtype=_torch.long).to(
                device
            )
            self._atomic_numbers = atomic_numbers

        # Create the input dictionary
        input_dict = {
            "ptr": self._ptr,
            "node_attrs": self._node_attrs,
            "batch": self._batch,
            "pbc": self._pbc,
            "positions": xyz_qm.to(self._dtype),
            "edge_index": edge_index,
            "shifts": shifts,
            "cell": self._cell,
        }

        # Get the in vacuo energy.
        EV_TO_HARTREE = 0.0367492929
        E_vac = self._mace(input_dict, compute_force=False)["interaction_energy"]

        assert (
            E_vac is not None
        ), "The model did not return any energy. Please check the input."

        E_vac = E_vac[0] * EV_TO_HARTREE

        # If there are no point charges, return the in vacuo energy and zeros
        # for the static and induced terms.
        if len(xyz_mm) == 0:
            zero = _torch.tensor(0.0, dtype=xyz_qm.dtype, device=device)
            return _torch.stack([E_vac, zero, zero])

        # Get the EMLE energy components.
        E_emle = self._emle(atomic_numbers, charges_mm, xyz_qm, xyz_mm)

        # Return the MACE and EMLE energy components.
        return _torch.stack([E_vac, E_emle[0], E_emle[1]])